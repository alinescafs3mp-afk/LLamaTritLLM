using System.Text.Json;
using TritStudio.Core;
if (args.Length == 2 && args[0] == "--worker") return await WorkerProtocolChecks.Run(args[1]);
var tests = new (string Name, Action Run)[]
{
    ("count includes norms and tied weights once", () => {
        foreach (var c in new[] { ModelConfig.Small, ModelConfig.Medium, ModelConfig.Large })
            Equal(c.ParameterCount, WeightLayout.For(c).Sum(s => (long)s.Count));
    }),
    ("configuration rejects invalid GQA", () => Throws(() => (ModelConfig.Small with { KvHeads = 3 }).Validate())),
    ("resource preflight rejects oversized budget request", () => Throws(() => (new ResourceOptions { MemoryMiB = 512, BatchSize = 32, SequenceLength = 512 }).Validate(ModelConfig.Large))),
    ("UTF8 count matches encoded tokens without array allocation", () => {
        foreach(string text in new[]{"", "plain", "кириллица", "🙂日本語", "a\nb"}) Equal(ByteTokenizer.Encode(text).Length,ByteTokenizer.TokenCount(text));
        Throws(()=>ByteTokenizer.TokenCount("\uD800"));
    }),
    ("UTF8 tokenizer round trip", () => {
        foreach (string s in new[] { "Привет, Дест!", "日本語", "🙂 кириллица", "line1\nline2", "\\\"" }) Equal(s, ByteTokenizer.Decode(ByteTokenizer.Encode(s)));
    }),
    ("UTF8 generation state rejects overlong and surrogate encodings", () => {
        var g = new Utf8Guard(); Assert(!g.Allows(0xC0 + 6)); g.Accept(0xED + 6); Assert(!g.Allows(0xA0 + 6));
        Assert(!g.Allows(ByteTokenizer.Eos)); g.Accept(0x9F + 6); g.Accept(0xBF + 6); Assert(g.Complete);
    }),
    ("prompt preserves latest message and full role pairs", () => {
        var p = ByteTokenizer.Prompt([new("old", "older", 0)], "latest", 32, 8);
        Equal(ByteTokenizer.Bos, p[0]); Equal(ByteTokenizer.Assistant, p[^1]); Assert(ByteTokenizer.Decode(p).EndsWith("latest"));
        Throws(() => ByteTokenizer.Prompt([], new string('a', 40), 32, 8));
    }),
    ("dialogue loss masks prompt and includes answer/EOS", () => {
        var e = Dataset.Encode(Dataset.Make("Q", "A"), 16);
        Equal(-100, e.Labels[0]); Equal(-100, e.Labels[3]); Equal('A' + 6, e.Labels[4]); Equal(ByteTokenizer.Eos, e.Labels[5]);
        Throws(() => Dataset.Encode(Dataset.Make(new string('a', 100), "b"), 16));
    }),
    ("text splitting preserves UTF8 scalar boundaries", () => {
        var parts = Dataset.SplitLongTexts([Dataset.Make(string.Concat(Enumerable.Repeat("🙂", 20)))], 16);
        Assert(parts.Length > 1); foreach (var p in parts) { _ = Dataset.Encode(p, 16); Assert(!p.Text.Contains('�')); }
    }),
    ("quantizer deterministic across worker counts", () => {
        var values = Enumerable.Range(0, 128).Select(i => MathF.Sin(i) * 0.1f).ToArray();
        var a = TernaryQuantizer.Unpack(TernaryQuantizer.Pack(values, 32, 2, 0.5f, 1), 128, 32);
        var b = TernaryQuantizer.Unpack(TernaryQuantizer.Pack(values, 32, 2, 0.5f, 3), 128, 32); Assert(a.SequenceEqual(b));
    }),
    ("packed file roundtrip and malformed length reject", () => Temp(dir => {
        var c = ModelConfig.Small with { Dimension = 16, HiddenDimension = 32, Layers = 1, Heads = 2, KvHeads = 1, GroupSize = 8, Context = 64 };
        var w = WeightSet.Initialize(c); string p = Path.Combine(dir, "model"); ModelFiles.Write(p, w, true);
        var q = ModelFiles.Read(p, true); var expected = TernaryQuantizer.Quantize(w);
        foreach (var s in WeightLayout.For(c)) Assert(q.Values[s.Name].SequenceEqual(expected.Values[s.Name]));
        File.AppendAllText(p, "bad"); Throws(() => ModelFiles.Read(p, true));
    })),
    ("replay idempotency, commit reconciliation, rollback", () => Temp(dir => {
        var r = new ReplayLedger(dir); var e = Dataset.Make("trusted fact"); Assert(r.Add(e)); Assert(!r.Add(e));
        var restarted = new ReplayLedger(dir); restarted.Reconcile([e.Id]); Equal(0, restarted.PendingCount);
        restarted.MarkRollback([]); Equal(0, restarted.Learned.Length); Assert(!restarted.Add(e));
    })),
    ("revision path cannot escape workspace", () => { Throws(() => ModelFiles.GetRevisionPath(".", "../r00000000000001")); }),
    ("sampler state resumes exactly", () => { var r = new SamplerRandom(42); r.Next(100); ulong saved = r.State; int x = r.Next(100); r.State = saved; Equal(x, r.Next(100)); }),
    ("sampling masks roles and respects top-k", () => {
        var x = Enumerable.Repeat(-100f, 262).ToArray(); x[ByteTokenizer.User] = 100; x[65 + 6] = 5;
        Equal(65 + 6, ManagedInference.Sample(x, [], new SamplingOptions(0, 1), new Random(1)));
    }),
    ("managed causal GQA forward matches independent numerical fixture", () => {
        var fixture = JsonData.Read<ReferenceFixture>(Path.Combine(AppContext.BaseDirectory, "reference.json"));
        var w = new WeightSet(fixture.Config, fixture.Weights); var session = new ManagedInference(TernaryQuantizer.Quantize(w), 0).NewSession();
        for (int i = 0; i < fixture.Tokens.Length; i++)
        {
            var y = session.Step(fixture.Tokens[i]);
            float error = y.Zip(fixture.Logits[i], (a, b) => MathF.Abs(a - b)).Max();
            Assert(error < 0.002f, $"position {i}: max error {error}");
        }
    }),
    ("dialogue context affects identity and masks all prior turns", () => {
        var e = Dataset.Make("name?", "Ada", history: new[] { new ChatTurn("Ada", "OK", 0) });
        Assert(e.Id != Dataset.Make("name?", "Ada").Id);
        var encoded = Dataset.Encode(e, 64);
        Equal(4, encoded.Labels.Count(x => x != -100));
        Assert(ByteTokenizer.Decode(encoded.Tokens).Contains("Ada"));
    }),
    ("learning context excludes private and other conversations", () => {
        var target = new ChatTurn("q", "a", 0, false, "one");
        ChatTurn[] history = [new("secret", "private", 0, true, "one"), new("foreign", "reply", 0, false, "other"), new("ok", "yes", 0, false, "one"), target];
        var kept = Dataset.LearningContext(history, target); Equal(1, kept.Length); Equal("ok", kept[0].User);
        Throws(() => Dataset.Make("q", "a", history: [history[0]]));
    }),
    ("invalid Unicode is rejected", () => { Throws(() => Dataset.Make("\uD800")); }),
    ("fixed tokenizer encodes new vocabulary without changing IDs", () => { int[] before = ByteTokenizer.Encode("A"); ByteTokenizer.Encode("невиданное слово"); Assert(before.SequenceEqual(ByteTokenizer.Encode("A"))); }),
    ("resource settings transfer to smaller CPU machines", () => { var r = new ResourceOptions { Threads = 128, BatchSize = 1, SequenceLength = 128 }; r.Validate(ModelConfig.Small); Assert(r.EffectiveThreads <= Environment.ProcessorCount); }),
    ("multi-turn JSONL reader validates role order", () => Temp(dir => {
        string path = Path.Combine(dir, "test.jsonl");
        File.WriteAllText(path, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hello\"},{\"role\":\"user\",\"content\":\"name?\"},{\"role\":\"assistant\",\"content\":\"Trit\"}]}");
        Equal(1, Dataset.Load(path)[0].History!.Length);
        File.WriteAllText(path, "{\"messages\":[{\"role\":\"system\",\"content\":\"bad\"},{\"role\":\"assistant\",\"content\":\"hello\"}]}"); Throws(() => Dataset.Load(path));
    })),
    ("evaluation splits cannot be imported as training data", () => {
        using var manifest = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"data","DATASET_MANIFEST.json")));
        foreach (var entry in manifest.RootElement.GetProperty("files").EnumerateObject())
            if (entry.Name is not ("seed" or "pretrain"))
                Throws(() => Dataset.LoadTraining(Path.Combine(AppContext.BaseDirectory, "data", entry.Name + ".jsonl")));
        Equal(manifest.RootElement.GetProperty("counts").GetProperty("seed").GetInt32(),BundledCorpus.Load(Path.Combine(AppContext.BaseDirectory, "data"), "seed.jsonl").Length);
        Equal(BundledCorpus.Version,manifest.RootElement.GetProperty("version").GetString());
    }),
    ("bundled dataset fits complete training sequences and has disjoint examples", () => {
        var all = new HashSet<string>();
        using var manifest = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"data","DATASET_MANIFEST.json")));
        foreach (var entry in manifest.RootElement.GetProperty("files").EnumerateObject()) {
            var examples = BundledCorpus.Load(Path.Combine(AppContext.BaseDirectory, "data"), entry.Name + ".jsonl");
            Equal(entry.Value.GetProperty("count").GetInt32(), examples.Length);
            foreach (var e in examples) { Assert(all.Add(e.Id), "Cross-split duplicate"); Assert(Dataset.FullSequenceLength(e) <= 512); _ = Dataset.Encode(e, 512); }
        }
    }),
    ("rollback reconciliation survives crash after pointer switch", () => Temp(dir => {
        var ledger = new ReplayLedger(dir); var a = Dataset.Make("old"); var b = Dataset.Make("new"); ledger.AddMany([a, b]); ledger.Finish([a.Id, b.Id], true);
        var restarted = new ReplayLedger(dir); restarted.Reconcile([a.Id]); Equal(1, restarted.Learned.Length); Equal(a.Id, restarted.Learned[0].Id);
    })),
    ("atomic replay add does not partly accept invalid duplicates", () => Temp(dir => {
        var ledger = new ReplayLedger(dir); var e = Dataset.Make("one"); Equal(1, ledger.AddMany([e, e])); Equal(0, ledger.AddMany([e])); Equal(1, new ReplayLedger(dir).PendingCount);
    })),
    ("quantizer rejects malformed arguments", () => { Throws(() => TernaryQuantizer.Pack([1f], 0, 1, 0.5f)); Throws(() => TernaryQuantizer.Pack([float.NaN], 1, 1, 0.5f)); }),


    ("prompt planning reports retained and dropped pairs exactly", () => {
        var history = new[] { new ChatTurn(new string('x', 40), "old", 0), new ChatTurn("near", "yes", 0) };
        var plan = ByteTokenizer.Plan(history, "now", 40, 8);
        Equal(1, plan.RetainedTurns); Equal(1, plan.DroppedTurns); Equal(3, plan.UserTokens);
        Assert(plan.Tokens.SequenceEqual(ByteTokenizer.Prompt(history, "now", 40, 8)));
        Assert(plan.Tokens.Length + plan.ReservedOutputTokens <= plan.Context);
    }),
    ("chat tokenizer rejects invalid Unicode instead of replacing it", () => Throws(() => ByteTokenizer.Encode("\uD800"))),
    ("invalid prompt reserve is rejected", () => { Throws(() => ByteTokenizer.Plan([], "x", 32, 0)); Throws(() => ByteTokenizer.Plan([], "x", 32, 32)); }),
    ("pending discard is durable and never removes learned examples", () => Temp(dir => {
        var ledger = new ReplayLedger(dir); var a = Dataset.Make("already learned"); var b = Dataset.Make("waiting");
        ledger.AddMany([a,b]); ledger.Finish([a.Id], true); Equal(1, ledger.DiscardPending());
        var loaded = new ReplayLedger(dir); Equal(0, loaded.PendingCount); Equal(1, loaded.Learned.Length); Equal(1, loaded.Summary.Discarded);
        Assert(!loaded.Add(b)); Equal(0, loaded.DiscardPending());
    })),
    ("discarded examples stay discarded during reconciliation", () => Temp(dir => {
        var ledger = new ReplayLedger(dir); var a = Dataset.Make("cancel me"); ledger.Add(a); ledger.DiscardPending(); ledger.Reconcile([]);
        Equal(0, ledger.PendingCount); Equal(1, ledger.Summary.Discarded);
    })),
    ("legacy train request enables explicit bundled-update default", () => {
        var request = JsonSerializer.Deserialize<TrainRequest>("{\"datasetPaths\":[],\"training\":{}}", JsonData.Options)!;
        Assert(request.IncludeBundledUpdates);
    }),
    ("legacy runtime flags remain readable", () => {
        var flags = JsonSerializer.Deserialize<RuntimeFlags>("{\"paused\":true,\"online\":false}", JsonData.Options)!;
        Assert(flags.Paused && flags.OnlineLearningRate is null);
    }),
    ("online learning rate survives runtime flag serialization", () => {
        var original = new RuntimeFlags(false, true, 0.000037);
        Equal(original, Protocol.Payload<RuntimeFlags>(Protocol.Element(original)));
    }),
    ("chat journal separates a crashed tail from the next record", () => Temp(dir => {
        string path = Path.Combine(dir,"chat.jsonl");
        var a = new ChatTurn("one","first",0); var b = new ChatTurn("two","second",1);
        ChatJournal.AppendAsync(path,a).GetAwaiter().GetResult();
        File.AppendAllText(path,"{\"user\":\"partial");
        ChatJournal.AppendAsync(path,b).GetAwaiter().GetResult();
        var read = ChatJournal.ReadTail(path); Equal(2,read.Turns.Length); Equal(1,read.SkippedRecords); Equal(b,read.Turns[1]);
    })),
    ("chat journal retains valid final record missing newline", () => Temp(dir => {
        string path = Path.Combine(dir,"chat.jsonl"); var a = new ChatTurn("a","b",0); var b = new ChatTurn("c","d",1);
        File.WriteAllText(path,JsonSerializer.Serialize(a,JsonData.Options)); ChatJournal.AppendAsync(path,b).GetAwaiter().GetResult();
        var read = ChatJournal.ReadTail(path); Equal(2,read.Turns.Length); Equal(0,read.SkippedRecords);
    })),
    ("chat journal tail read is bounded", () => Temp(dir => {
        string path = Path.Combine(dir,"chat.jsonl");
        File.WriteAllLines(path, Enumerable.Range(0,1000).Select(i => JsonSerializer.Serialize(new ChatTurn("строка "+i,"ответ",i),JsonData.Options)));
        var read = ChatJournal.ReadTail(path,limit:5,maxBytes:2048); Equal(5,read.Turns.Length); Equal(999L,read.Turns[^1].Revision); Assert(read.BytesRead <= 2048);
    })),
    ("aggregate dataset import checks duplicate files and limit", () => Temp(dir => {
        string path = Path.Combine(dir,"one.txt"); File.WriteAllText(path,"hello");
        Equal(1,Dataset.LoadManyTraining([path,path]).Length);
        Throws(() => Dataset.LoadManyTraining(Enumerable.Range(0,33).Select(i=>Path.Combine(dir,i+".txt"))));
    })),
    ("dataset preparation honors cancellation before work", () => {
        using var stop = new CancellationTokenSource(); stop.Cancel();
        Throws(() => Dataset.SplitLongTexts([Dataset.Make("hello")],32,stop.Token));
        Throws(() => Dataset.EncodeAll([Dataset.Make("hello")],32,1,stop.Token));
    }),
    ("bundled corpus verifies all manifests including frozen tests", () => {
        string root = Path.Combine(AppContext.BaseDirectory,"data");
        foreach (string name in new[] {"seed.jsonl","validation.jsonl","test.jsonl","challenge.jsonl","challenge-v4.jsonl","challenge-v5.jsonl","challenge-v6.jsonl","challenge-v7.jsonl","challenge-v8.jsonl","pretrain.jsonl"}) Assert(BundledCorpus.Load(root,name).Length > 0);
        Throws(() => BundledCorpus.Load(root,"../secret.jsonl"));
    }),
    ("corrupted bundled corpus is rejected before training", () => Temp(dir => {
        string source = Path.Combine(AppContext.BaseDirectory,"data");
        File.Copy(Path.Combine(source,"DATASET_MANIFEST.json"),Path.Combine(dir,"DATASET_MANIFEST.json"));
        File.Copy(Path.Combine(source,"seed.jsonl"),Path.Combine(dir,"seed.jsonl")); File.AppendAllText(Path.Combine(dir,"seed.jsonl")," ");
        Throws(() => BundledCorpus.Load(dir,"seed.jsonl"));
    })),
    ("snapshot pruning follows active ancestry instead of newer abandoned branches", () => Temp(dir => {
        for (int i=0;i<10;i++) FakeRevision(dir,i,i==0?null:i-1);
        JsonData.AtomicWrite(Path.Combine(dir,"active.json"),new ActiveRevision("r0000000000000003"));
        var preview = WorkspaceMaintenance.Prune(dir,keep:3); Assert(!preview.Applied); Equal(7,preview.Removed.Length);
        Assert(Directory.Exists(ModelFiles.GetRevisionPath(dir,"r0000000000000009")));
        var result = WorkspaceMaintenance.Prune(dir,keep:3,apply:true); Equal(3,result.Kept);
        foreach (int i in new[] {1,2,3}) Assert(Directory.Exists(ModelFiles.GetRevisionPath(dir,$"r{i:D16}")));
        Assert(!Directory.Exists(ModelFiles.GetRevisionPath(dir,"r0000000000000009")));
    })),
    ("revision IDs do not recycle after branch pruning", () => Temp(dir => {
        for (int i=0;i<10;i++) FakeRevision(dir,i,i==0?null:i-1);
        JsonData.AtomicWrite(Path.Combine(dir,"active.json"),new ActiveRevision("r0000000000000003"));
        WorkspaceMaintenance.Prune(dir,keep:3,apply:true);
        Equal(10L,RevisionSequence.Reserve(dir)); Equal(11L,RevisionSequence.Reserve(dir));
    })),
    ("revision counter rejects malformed state", () => Temp(dir => {
        JsonData.AtomicWrite(Path.Combine(dir,"revision-counter.json"),-2L); Throws(()=>RevisionSequence.Reserve(dir));
    })),
    ("GUI-owned maintenance keeps UI lease and requires trainer exit", () => Temp(dir => {
        FakeRevision(dir,0,null); JsonData.AtomicWrite(Path.Combine(dir,"active.json"),new ActiveRevision("r0000000000000000"));
        using var ui = new FileStream(Path.Combine(dir,".ui.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None);
        using (var worker = new FileStream(Path.Combine(dir,".trainer.lock"),FileMode.OpenOrCreate,FileAccess.ReadWrite,FileShare.None))
            Throws(() => WorkspaceMaintenance.PruneWithUiLease(dir,ui,keep:2));
        Equal(1,WorkspaceMaintenance.PruneWithUiLease(dir,ui,keep:2).Kept); Assert(ui.CanWrite);
    })),


    ("batch planner preserves target count and never truncates a target", () => {
        var corpus = Enumerable.Range(0, 64).Select(i => Example(i, 8 + i * 3)).ToArray();
        var planner = new BatchPlanner(corpus); var rng = new SamplerRandom(17);
        for (int n = 0; n < 100; n++) {
            var batch = planner.Select(8,rng);
            Equal(batch.Length,batch.Examples.Max(BatchPlanner.EffectiveLength));
            Equal(batch.Examples.Sum(x => (long)x.Labels.Count(v => v != -100)),batch.TargetTokens);
            Assert(batch.InputTokens <= batch.PaddedPositions && batch.PaddingFraction is >= 0 and < 1);
        }
    }),
    ("length bucketing reduces padding without weighting small buckets equally", () => {
        var corpus = Enumerable.Range(0,100).Select(i => Example(i,i<90?16:240)).ToArray();
        var planner = new BatchPlanner(corpus); var a = new SamplerRandom(42); var b = new SamplerRandom(42);
        long bucketPadding=0,flatPadding=0,total=0,longRows=0;
        for (int i=0;i<10000;i++) {
            var x=planner.Select(8,a); var y=planner.Select(8,b,false);
            bucketPadding+=x.PaddedPositions-x.InputTokens; flatPadding+=y.PaddedPositions-y.InputTokens;
            total+=x.Examples.Length; longRows+=x.Examples.Count(e=>e.Tokens.Length>32);
        }
        Assert(bucketPadding<flatPadding); Assert((double)longRows/total is > 0.08 and < 0.12,"Uneven buckets biased marginal sampling.");
    }),
    ("bucket sampler resumes from its saved RNG state", () => {
        var corpus=Enumerable.Range(0,64).Select(i=>Example(i,10+i*2)).ToArray(); var p=new BatchPlanner(corpus); var r=new SamplerRandom(27);
        p.Select(4,r); ulong state=r.State; var expected=p.Select(4,r).Examples.Select(x=>x.Id).ToArray();
        var restored=new SamplerRandom(state); Assert(expected.SequenceEqual(new BatchPlanner(corpus).Select(4,restored).Examples.Select(x=>x.Id)));
    }),
    ("required online correction stays in mixed replay batch", () => {
        var corpus=Enumerable.Range(0,64).Select(i=>Example(i,16)).ToArray(); var correction=Example(999,240);
        var b=new BatchPlanner(corpus).Select(4,new SamplerRandom(42),true,correction);
        Equal(correction.Id,b.Examples[0].Id); Equal(240,b.Length); Assert(b.Examples.Skip(1).All(x=>x.Tokens.Length==16));
    }),
    ("batch planner rejects malformed labels and empty supervision", () => {
        Throws(()=>new BatchPlanner([])); Throws(()=>new BatchPlanner([new([1],[1,2],"bad")]));
        Throws(()=>new BatchPlanner([new([1],[-100],"bad")]));
    }),
    ("latest-value mailbox has bounded scheduling and suppresses stale completion", () => {
        var queue=new Queue<Action>(); var seen=new List<int>(); using var box=new LatestValueMailbox<int>(queue.Enqueue,seen.Add);
        for(int i=0;i<10000;i++) box.Post(i); Equal(1,queue.Count); queue.Dequeue()(); Equal(9999,seen.Single());
        box.Post(10001); Equal(1,queue.Count); box.Dispose(); queue.Dequeue()(); Equal(1,seen.Count);
        box.Post(10002); Equal(0,queue.Count);
    }),
    ("latest-value mailbox accepts an update published during consumption", () => {
        var queue=new Queue<Action>(); var seen=new List<int>(); LatestValueMailbox<int>? box=null;
        using var owned=box=new LatestValueMailbox<int>(queue.Enqueue,x=>{seen.Add(x);if(x==1)box!.Post(2);});
        owned.Post(1); queue.Dequeue()(); Equal(1,queue.Count); queue.Dequeue()(); Assert(seen.SequenceEqual(new[]{1,2}));
    }),
    ("export protects entire workspace but accepts a sibling directory", () => Temp(dir => {
        string workspace=Path.Combine(dir,"work"); Directory.CreateDirectory(workspace); string src=Path.Combine(workspace,"model.tritmodel");
        foreach(var name in new[]{"active.json","replay.json","settings.json","nested/model.tritmodel"})
            Throws(()=>ExportGuard.Validate(src,Path.Combine(workspace,name),workspace));
        Throws(()=>ExportGuard.Validate(src,src,null));
        ExportGuard.Validate(src,Path.Combine(dir,"work-other","model.tritmodel"),workspace);
        Throws(()=>ExportGuard.Validate(src,Path.Combine(dir,"any.tritmodel"),Path.GetPathRoot(dir)));
    })),
    ("export refuses linked ancestor on systems that permit symbolic links", () => Temp(dir => {
        if(OperatingSystem.IsWindows())return; string source=Path.Combine(dir,"source");Directory.CreateDirectory(source);
        string link=Path.Combine(dir,"linked");Directory.CreateSymbolicLink(link,source);
        Throws(()=>ExportGuard.Validate(Path.Combine(source,"a"),Path.Combine(link,"b"),null));
    })),
    ("JSON write bound preserves previous valid state and removes staging", () => Temp(dir => {
        string file=Path.Combine(dir,"state.json");JsonData.AtomicWrite(file,new[]{1,2},32);
        Throws(()=>JsonData.AtomicWrite(file,new string('x',1000),32)); Assert(JsonData.Read<int[]>(file,32).SequenceEqual(new[]{1,2}));
        Equal(1,Directory.GetFiles(dir).Length); Throws(()=>JsonData.Read<int[]>(file,1));
        Throws(()=>JsonData.AtomicWrite(file,1,JsonData.MaxStateBytes+1));
    })),
    ("replay cached counters match mutations and no-op reconciliation does not write", () => Temp(dir => {
        var ledger=new ReplayLedger(dir);var a=Dataset.Make("first");var b=Dataset.Make("second");ledger.AddMany([a,b]);
        Equal(2,ledger.Summary.Pending);Equal(a.Id,ledger.PeekPending(1).Single().Id);
        ledger.Finish([a.Id],true);Equal(1,ledger.Summary.Learned);Equal(a.Id,ledger.RecentLearned(1).Single().Id);
        string file=Path.Combine(dir,"replay.json");var fixedTime=new DateTime(2020,1,1,0,0,0,DateTimeKind.Utc);File.SetLastWriteTimeUtc(file,fixedTime);
        ledger.Reconcile([a.Id]);Equal(fixedTime,File.GetLastWriteTimeUtc(file));
        ledger.DiscardPending();Equal(0,ledger.PendingCount);Equal(1,ledger.Summary.Discarded);Equal(1,new ReplayLedger(dir).Summary.Learned);
    })),
    ("invalid replay IDs cannot be persisted", () => Temp(dir => {
        var ledger=new ReplayLedger(dir);var e=Dataset.Make("first") with {Id="wrong"};Throws(()=>ledger.Add(e));Equal(0,ledger.PendingCount);
        JsonData.AtomicWrite(Path.Combine(dir,"replay.json"),new ReplayDocument([new(Dataset.Make("ok"),"unknown")]));Throws(()=>new ReplayLedger(dir));
    })),
    ("inference verifies packed file without reading training-only files", () => Temp(dir => {
        string path=ModelFiles.GetRevisionPath(dir,"r0000000000000000");Directory.CreateDirectory(path);
        var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=64};
        ModelFiles.Write(Path.Combine(path,"model.tritmodel"),WeightSet.Initialize(c),true);
        var info=new RevisionInfo(0,0,0,null,"test",DateTimeOffset.UtcNow,"unused",ModelFiles.Hash(Path.Combine(path,"model.tritmodel")),null,"unused","unused","unused",0,null);
        JsonData.AtomicWrite(Path.Combine(path,"revision.json"),info);
        Equal(c,ModelFiles.ReadInferenceRevision(path).Weights.Config);Throws(()=>ModelFiles.VerifyRevision(path));
        File.AppendAllText(Path.Combine(path,"model.tritmodel"),"corruption");Throws(()=>ModelFiles.ReadInferenceRevision(path));
    })),
    ("invalid snapshot metadata is rejected before large model parsing", () => Temp(dir => {
        string path=ModelFiles.GetRevisionPath(dir,"r0000000000000000");Directory.CreateDirectory(path);
        var info=new RevisionInfo(0,-1,0,null,"test",DateTimeOffset.UtcNow,"","",null,"","","",0,null);
        JsonData.AtomicWrite(Path.Combine(path,"revision.json"),info);Throws(()=>ModelFiles.ReadRevisionInfo(path));
        JsonData.AtomicWrite(Path.Combine(path,"revision.json"),info with {Step=0,Parent=0});Throws(()=>ModelFiles.ReadRevisionInfo(path));
    })),
    ("quantizer parallel branch matches serial branch on larger tensors", () => {
        var values=Enumerable.Range(0,65536).Select(i=>MathF.Sin(i)*0.1f).ToArray();
        var a=TernaryQuantizer.Pack(values,32,2,0.5f,1);var b=TernaryQuantizer.Pack(values,32,2,0.5f,4);
        for(int p=0;p<a.Length;p++){Assert(a[p].Scales.SequenceEqual(b[p].Scales));Assert(a[p].Trits.SequenceEqual(b[p].Trits));}
    }),
    ("unpacker rejects invalid shapes before integer division", () => {
        Throws(()=>TernaryQuantizer.Unpack([],8,0));Throws(()=>TernaryQuantizer.Unpack([],8,4));
        Throws(()=>TernaryQuantizer.Unpack([new([1f],[0])],7,4));
        Throws(()=>TernaryQuantizer.Pack([float.MaxValue,float.MaxValue],2,1,0.5f));
    }),
    ("sampling rejects nonfinite values instead of generating corrupt text", () => {
        var x=new float[262];x[100]=float.NaN;Throws(()=>ManagedInference.Sample(x,[],new(),new Random(1)));
        x[100]=float.PositiveInfinity;Throws(()=>ManagedInference.Sample(x,[],new(),new Random(1)));
    }),
    ("performance options are backward compatible and persist opt-outs", () => {
        var defaults=JsonSerializer.Deserialize<ResourceOptions>("{}",JsonData.Options)!;Assert(defaults.UseSdpa&&defaults.BucketByLength);
        var opt=new ResourceOptions{UseSdpa=false,BucketByLength=false};
        Equal(opt,JsonSerializer.Deserialize<ResourceOptions>(JsonSerializer.Serialize(opt,JsonData.Options),JsonData.Options)!);
    }),

};
tests = tests.Concat(Audit5Checks.All).Concat(Audit6Checks.All).Concat(Audit7Checks.All).Concat(Audit8Checks.All).Concat(Audit9Checks.All).Concat(Audit10Checks.All).Concat(Audit11Checks.All).Concat(Audit12Checks.All).Concat(Audit13Checks.All).Concat(Audit14Checks.All).Concat(Audit15Checks.All).Concat(Audit16Checks.All).ToArray();
int failures = 0;
foreach (var t in tests) try { t.Run(); Console.WriteLine("PASS " + t.Name); } catch (Exception e) { failures++; Console.Error.WriteLine("FAIL " + t.Name + ": " + e); }
Console.WriteLine($"{tests.Length - failures}/{tests.Length} passed"); return failures == 0 ? 0 : 1;
static void Assert(bool value, string? message = null) { if (!value) throw new Exception(message ?? "Assertion failed"); }
static void Equal<T>(T expected, T actual) { if (!EqualityComparer<T>.Default.Equals(expected, actual)) throw new Exception($"Expected {expected}; got {actual}"); }
static void Throws(Action f) { try { f(); } catch { return; } throw new Exception("Expected an exception"); }
static void Temp(Action<string> f) { string d = Path.Combine(Path.GetTempPath(), "trit-tests-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(d); try { f(d); } finally { Directory.Delete(d, true); } }

// Metadata-only fixtures for maintenance tests. These are NOT executable neural network weights.
static void FakeRevision(string root, long id, long? parent)
{
    string path = ModelFiles.GetRevisionPath(root,$"r{id:D16}"); Directory.CreateDirectory(path);
    foreach (string name in new[] {"master.weights","model.tritmodel","optimizer.bin","state.json","commit.json"}) File.WriteAllText(Path.Combine(path,name),name+id);
    string Hash(string name) => ModelFiles.Hash(Path.Combine(path,name));
    var info = new RevisionInfo(id,id,0,null,"maintenance fixture",DateTimeOffset.UtcNow,Hash("master.weights"),Hash("model.tritmodel"),parent,Hash("optimizer.bin"),Hash("state.json"),Hash("commit.json"),0,null);
    JsonData.AtomicWrite(Path.Combine(path,"revision.json"),info);
}

static EncodedExample Example(int id,int length) => new(Enumerable.Repeat(10,length).ToArray(),Enumerable.Repeat(10,length).ToArray(),id.ToString());

public sealed record ReferenceFixture(ModelConfig Config, Dictionary<string, float[]> Weights, int[] Tokens, float[][] Logits);
