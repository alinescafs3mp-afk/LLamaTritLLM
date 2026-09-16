using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

public static class TrainerSelfTest
{
    public static int Run(bool requireCuda)
    {
        string root = Path.Combine(Path.GetTempPath(), "trit-training-test-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(root);
        try
        {
            torch.set_num_interop_threads(1);
            if (requireCuda && !torch.cuda.is_available()) throw new InvalidOperationException("CUDA requested but unavailable. This is a failure, not a CPU test pass.");
            var c = new ModelConfig { Dimension = 16, HiddenDimension = 32, Layers = 1, Heads = 2, KvHeads = 1, Context = 64, GroupSize = 8 };
            var resources = new ResourceOptions { Threads = 1, MemoryMiB = 4096, BatchSize = 2, SequenceLength = 64, PreferCuda = requireCuda };
            var options = new TrainingOptions { LearningRate = 0.001 };
            var initial = WeightSet.Initialize(c);
            CheckSiluParity(requireCuda);
            CheckAttentionParity(initial, requireCuda);
            CheckSelectedTargetParity(initial, requireCuda);
            CheckGradientNorm(initial, requireCuda);
            using var training = new TrainingSession(initial, resources, options);
            var firstCapture = training.CapturePublicationMaster();
            if (!ReferenceEquals(firstCapture, initial) || training.InitialMasterReuses != 1)
                throw new Exception("Untrained publication did not reuse the immutable initial master.");
            var nativeInitial = training.Model.CopyMaster();
            foreach (var shape in WeightLayout.For(c))
                if (!firstCapture.Values[shape.Name].SequenceEqual(nativeInitial.Values[shape.Name]))
                    throw new Exception("Initial publication differs from native FP32 weights.");
            using (var cancelled = new CancellationTokenSource())
            {
                cancelled.Cancel(); bool stopped = false;
                try { training.CapturePublicationMaster(cancelled.Token); } catch (OperationCanceledException) { stopped = true; }
                if (!stopped) throw new Exception("Initial publication ignored cancellation.");
            }
            using (var changed = new TrainingSession(initial, resources, options))
            {
                changed.Model.MarkUpdated();
                if (ReferenceEquals(initial, changed.CapturePublicationMaster()))
                    throw new Exception("Initial publication ignored model version invalidation.");
            }
            using (var restored = new TrainingSession(initial, resources, options))
            {
                string zeroOptimizer = Path.Combine(root, "initial-optimizer.bin");
                var zeroState = restored.State; restored.SaveOptimizer(zeroOptimizer);
                restored.RestoreState(zeroOptimizer, zeroState);
                if (ReferenceEquals(initial, restored.CapturePublicationMaster()))
                    throw new Exception("State restoration left the initial-reference shortcut enabled.");
                bool failed = false;
                try { restored.RestoreState(Path.Combine(root, "absent-optimizer.bin"), zeroState); }
                catch (IOException) { failed = true; }
                if (!failed || ReferenceEquals(initial, restored.CapturePublicationMaster()))
                    throw new Exception("Failed restore reused unverified initial-reference state.");
            }
            using (var scope = torch.NewDisposeScope())
            {
                int[] ids = [ByteTokenizer.Bos, ByteTokenizer.User, 'q' + 6, ByteTokenizer.Eos, ByteTokenizer.Assistant];
                var input = torch.tensor(ids.Select(x => (long)x).ToArray(), dtype: torch.ScalarType.Int64, device: training.Model.Device).reshape(1, -1);
                var torchOutput = training.Model.Forward(input).detach().cpu().data<float>().ToArray();
                var cpu = new ManagedInference(TernaryQuantizer.Quantize(initial), 0).NewSession();
                for (int pos = 0; pos < ids.Length; pos++)
                {
                    var result = cpu.Step(ids[pos]); float max = 0;
                    for (int i = 0; i < result.Length; i++) max = MathF.Max(max, MathF.Abs(result[i] - torchOutput[pos * result.Length + i]));
                    if (max > 0.002) throw new Exception($"CPU/Torch logits mismatch at {pos}: {max}");
                }
            }
            var examples = Dataset.EncodeAll([Dataset.Make("q", "a"), Dataset.Make("x", "a")], 64, 1);
            long quantizedMatrices = WeightLayout.For(c).Count(x => x.Quantized);
            var validation = Enumerable.Range(0, 9).Select(i => Dataset.Encode(Dataset.Make("q" + i, "a"), 64)).ToArray();
            long builds = training.Model.QuantizationBuilds;
            double check = training.Evaluate(validation);
            if (training.Model.QuantizationBuilds - builds != quantizedMatrices) throw new Exception("Validation re-quantized matrices per batch.");
            long passes = training.ValidationPasses;
            if (training.Evaluate(validation) != check || training.ValidationPasses != passes || training.ValidationCacheHits != 1)
                throw new Exception("Unchanged validation failed to reuse its exact result.");
            // A different corpus reference must execute a fresh pass, even if its content is equal.
            double fresh = training.Evaluate(validation.ToArray());
            if (Math.Abs(fresh - check) > 1e-6) throw new Exception("Cached evaluation changed loss.");
            var invalid = new[] { new EncodedExample(new int[c.Context + 1], Enumerable.Repeat(2, c.Context + 1).ToArray(), "invalid") };
            bool rejected = false; try { training.Evaluate(invalid); } catch (ArgumentException) { rejected = true; }
            if (!rejected || Math.Abs(training.Evaluate(validation) - check) > 1e-6) throw new Exception("Failed evaluation poisoned a subsequent corpus.");
            double before = training.Evaluate(examples);
            long beforePasses = training.ValidationPasses;
            long preparedHits = training.ValidationBatchCacheHits, preparedBytes = training.ValidationPreparedBytes;
            training.TrainStep(examples, options.LearningRate, CancellationToken.None);
            if (ReferenceEquals(initial, training.CapturePublicationMaster())) throw new Exception("Updated model reused random initial weights.");
            training.Evaluate(examples);
            if (preparedBytes <= 0 || training.ValidationPreparedBytes != preparedBytes || training.ValidationBatchCacheHits <= preparedHits)
                throw new Exception("Immutable validation preparation was not reused across a real weight update.");
            if (training.ValidationPasses != beforePasses + 1) throw new Exception("Weight update did not invalidate validation result.");
            for (int i = 0; i < 32; i++) training.TrainStep(examples, options.LearningRate, CancellationToken.None);
            double after = training.Evaluate(examples);
            if (initial.CountChanged(training.Model.CopyMaster()) == 0 || !(after < before)) throw new Exception($"No demonstrated fitting: {before} -> {after}");
            var master = training.Model.CopyMaster(); var state = training.State;
            string optimizer = Path.Combine(root, "optimizer.bin"); training.SaveOptimizer(optimizer);
            var bucketCorpus = Dataset.EncodeAll(Enumerable.Range(0, 64).Select(i => Dataset.Make("q" + i + new string('x', i % 20), "a")).ToArray(), 64, 1);
            training.TrainStep(bucketCorpus, options.LearningRate, CancellationToken.None); var expected = training.Model.CopyMaster();
            training.Restore(master, optimizer, state); training.TrainStep(bucketCorpus, options.LearningRate, CancellationToken.None);
            var actual = training.Model.CopyMaster();
            foreach (var s in WeightLayout.For(c))
            {
                float error = expected.Values[s.Name].Zip(actual.Values[s.Name], (a, b) => MathF.Abs(a - b)).Max();
                if (error > (requireCuda ? 0.0001 : 0.00001)) throw new Exception($"Optimizer/sampler resume mismatch: {s.Name}, {error}");
            }
            CheckTrainedExportParity(training, root);
            CheckSnapshotPublication(training, root, resources, options);
            Console.WriteLine($"PASS real gradients, loss fitting ({before:F4} -> {after:F4}), managed/Torch parity, full-state resume on {training.DeviceName}.");
            Console.WriteLine("PASS SDPA/explicit forward+backward, versioned evaluation cache, failure recovery, bucketed sampler resume.");
            Console.WriteLine("This is a numerical smoke test, not a model quality benchmark."); return 0;
        }
        catch (Exception e) { Console.Error.WriteLine("FAIL " + e); return 1; }
        finally { Directory.Delete(root, true); }
    }

    // Owner reported poor text after direct conversational training. Verify the actual trained
    // device -> FP32 master -> packed disk -> managed autoregressive path, not just initial weights.
    // This is numerical transport parity, NOT proof of conversational competence or a fluency test.
    private static void CheckTrainedExportParity(TrainingSession training, string root)
    {
        var master = training.CapturePublicationMaster();
        if (training.Step <= 0) throw new Exception("Trained export fixture has not changed any weights.");
        string packedPath = Path.Combine(root, "trained-export.tritmodel");
        ModelFiles.Write(packedPath, master, true);
        var loaded = ModelFiles.Read(packedPath, true);
        var quantized = TernaryQuantizer.Quantize(master);
        foreach (var shape in WeightLayout.For(master.Config))
            if (!quantized.Values[shape.Name].SequenceEqual(loaded.Values[shape.Name]))
                throw new Exception("Packed trained weights differ from the in-memory quantizer: " + shape.Name);
        foreach (string prompt in new[] { "q", "x", "привет" })
        {
            int[] ids = ByteTokenizer.Prompt([], prompt, master.Config.Context, 16);
            using var scope = torch.NewDisposeScope();
            using var noGrad = torch.no_grad();
            var tensor = torch.tensor(ids.Select(x => (long)x).ToArray(), dtype: torch.ScalarType.Int64,
                device: training.Model.Device).reshape(1, -1);
            var expected = training.Model.Forward(tensor).detach().cpu().contiguous().data<float>().ToArray();
            var cpu = new ManagedInference(loaded, 0, 1).NewSession();
            for (int pos = 0; pos < ids.Length; pos++)
            {
                var actual = cpu.Step(ids[pos]);
                for (int i = 0; i < actual.Length; i++)
                {
                    float wanted = expected[pos * actual.Length + i];
                    float tolerance = 0.002f + 0.0002f * MathF.Abs(wanted);
                    if (!float.IsFinite(wanted) || !float.IsFinite(actual[i]) || MathF.Abs(wanted - actual[i]) > tolerance)
                        throw new Exception($"Trained native/packed managed mismatch: pos={pos}, token={i}, expected={wanted}, actual={actual[i]}");
                }
            }
            var prefixModel = new ManagedInference(loaded, 0, 1);
            var reference = prefixModel.NewSession(optimizePrefix: false);
            var optimized = prefixModel.NewSession();
            float[] refLast = [], fastLast = [];
            for (int p = 0; p < ids.Length; p++)
            {
                refLast = reference.Step(ids[p], computeLogits: p == ids.Length - 1);
                fastLast = optimized.Step(ids[p], computeLogits: p == ids.Length - 1);
            }
            CheckPrefix(refLast, fastLast);
            for (int p = 0; p < Math.Min(3, master.Config.Context - ids.Length); p++)
                CheckPrefix(reference.Step(30+p), optimized.Step(30+p));
            if (optimized.PrefixFinalBlockSkips != ids.Length-1) throw new Exception("Trained prefix shortcut did not run.");
            void CheckPrefix(float[] a, float[] b)
            {
                if (a.Length != b.Length) throw new Exception("Trained prefix length mismatch.");
                for(int i=0;i<a.Length;i++)
                    if(!float.IsFinite(a[i])||!float.IsFinite(b[i])||MathF.Abs(a[i]-b[i])>2e-5f)
                        throw new Exception("Trained prefix cache differs from full last block.");
            }
            var sampling = new SamplingOptions(Temperature: 0, MaxNewTokens: 16);
            string a = new ManagedInference(quantized, 0, 1).Generate(ids, () => sampling, null, CancellationToken.None);
            string b = new ManagedInference(loaded, 0, 1).Generate(ids, () => sampling, null, CancellationToken.None);
            if (a != b) throw new Exception("Packed round-trip changed the deterministic trained response.");
        }
        Console.WriteLine("PASS trained device/master/packed-file/managed incremental logits and deterministic round-trip (not fluency).");
    }

    private static void CheckSiluParity(bool cuda)
    {
        using var scope = torch.NewDisposeScope(); var device = cuda ? torch.CUDA : torch.CPU;
        var values=Enumerable.Range(0,257).Select(i=>(i-128)/8f).ToArray();
        var left=torch.nn.Parameter(torch.tensor(values,device:device));
        var right=torch.nn.Parameter(torch.tensor(values,device:device));
        var reference=left*left.sigmoid(); var fused=torch.nn.functional.silu(right);
        reference.sum().backward(); fused.sum().backward();
        float valueError=(reference-fused).abs().max().item<float>();
        float gradError=(left.grad!-right.grad!).abs().max().item<float>();
        if(!float.IsFinite(valueError)||!float.IsFinite(gradError)||valueError>1e-5||gradError>1e-5)
            throw new Exception($"SiLU parity failed: values={valueError}, gradients={gradError}");
        Console.WriteLine("PASS native SiLU/reference forward and gradient parity.");
    }
    private static void CheckGradientNorm(WeightSet initial, bool cuda)
    {
        using var scope = torch.NewDisposeScope(); var device = cuda ? torch.CUDA : torch.CPU;
        using var model = new TorchModel(initial, device);
        var ids = torch.tensor(new long[]{1,3,65,2,4}, dtype:torch.ScalarType.Int64,device:device).reshape(1,-1);
        model.Forward(ids).pow(2).mean().backward();
        var old = torch.zeros(Array.Empty<long>(),dtype:torch.ScalarType.Float32,device:device);
        foreach(var p in model.Parameters) if(p.grad is { } g) old = old + g.pow(2).sum();
        var current = TrainingSession.SquaredGradientNorm(model.Parameters, device);
        double error=(old-current).abs().item<float>();
        if(!double.IsFinite(error)||error>1e-6+1e-5*old.item<float>())throw new Exception("Scoped gradient norm differs from reference.");
        foreach(var p in model.Parameters) if(p.grad is not { } g || !double.IsFinite(g.abs().sum().item<float>()))throw new Exception("Norm calculation disposed a live gradient.");
        Console.WriteLine("PASS scoped gradient norm/reference parity; parameter gradients remain usable.");
    }
    private static void CheckSnapshotPublication(TrainingSession session,string root,ResourceOptions resources,TrainingOptions options)
    {
        string workspace=Path.Combine(root,"publication"); using var store=new WorkspaceStore(workspace);
        var master=session.Model.CopyMaster(); var settings=new WorkspaceSettings(master.Config,resources,options);
        TrainingExample[] train=[Dataset.Make("q","a")],validation=[Dataset.Make("z","a")];
        var first=store.Publish(session,master,1,"fixture",[],null,2,settings,train,validation);
        string active=ModelFiles.ActivePath(workspace)!;var info=ModelFiles.VerifyRevision(active);
        if(info.Revision!=first.Event.Info.Revision||ModelFiles.ReadInferenceRevision(active,2).Info!=info)throw new Exception("Hashed publication did not verify.");
        using var cancelled=new CancellationTokenSource();cancelled.Cancel();bool rejected=false;
        try{store.Publish(session,master,1,"cancelled",[],info.Revision,2,settings,train,validation,cancelled.Token);}catch(OperationCanceledException){rejected=true;}
        if(!rejected||ModelFiles.ActivePath(workspace)!=active||Directory.GetDirectories(Path.Combine(workspace,"revisions"),".stage-*").Length!=0)
            throw new Exception("Cancelled publication changed active snapshot or leaked staging.");
        bool copyCancelled=false;try{session.Model.CopyMaster(cancelled.Token);}catch(OperationCanceledException){copyCancelled=true;}
        if(!copyCancelled)throw new Exception("Master copy ignored pre-cancellation.");
        session.Restore(ModelFiles.Read(Path.Combine(active,"master.weights"),false),Path.Combine(active,"optimizer.bin"),JsonData.Read<TrainerState>(Path.Combine(active,"state.json")));
        if(master.CountChanged(session.Model.CopyMaster())!=0)throw new Exception("Published master did not restore.");
        Console.WriteLine("PASS hashed publication/full verification/cancellation/optimizer restore.");
    }
    private static void CheckSelectedTargetParity(WeightSet initial, bool cuda)
    {
        var device = cuda ? torch.CUDA : torch.CPU;
        foreach (bool sdpa in new[] { false, true })
        {
            using var scope = torch.NewDisposeScope();
            using var full = new TorchModel(initial, device, sdpa);
            using var selected = new TorchModel(initial, device, sdpa);
            const int b = 2, t = 13;
            var ids = torch.tensor(Enumerable.Range(0,b*t).Select(i=>(long)(6+i%100)).ToArray(),dtype:torch.ScalarType.Int64,device:device).reshape(b,t);
            long[] positions = [4,5,6,12,20,25]; long[] labels = [30,31,32,2,45,2];
            var at = torch.tensor(positions,dtype:torch.ScalarType.Int64,device:device);
            var targets = torch.tensor(labels,dtype:torch.ScalarType.Int64,device:device);
            var fullLogits = full.Forward(ids).reshape(-1,ByteTokenizer.VocabularySize).index_select(0,at);
            var compactLogits = selected.Forward(ids,at);
            double delta = (fullLogits-compactLogits).abs().max().item<float>();
            if(!double.IsFinite(delta)||delta>0.002)throw new Exception("Selected output logits mismatch: "+delta);
            var a = torch.nn.functional.cross_entropy(fullLogits,targets);
            var z = torch.nn.functional.cross_entropy(compactLogits,targets);
            if(Math.Abs(a.item<float>()-z.item<float>())>0.002)throw new Exception("Selected output loss differs.");
            a.backward();z.backward();
            for(int i=0;i<full.Parameters.Count;i++)
            {
                var ga=full.Parameters[i].grad ?? throw new Exception("Missing full gradient");
                var gz=selected.Parameters[i].grad ?? throw new Exception("Missing selected gradient");
                double error=(ga-gz).abs().max().item<float>();
                double magnitude=ga.abs().max().item<float>();
                if(!double.IsFinite(error)||error>0.0001+0.001*magnitude)throw new Exception("Selected target gradient mismatch: "+i+" "+error);
            }
            if(initial.CountChanged(full.CopyMaster())!=0||initial.CountChanged(selected.CopyMaster())!=0)
                throw new Exception("Forward/quantization mutated master weights.");
        }
        Console.WriteLine("PASS selected/full output loss+gradient parity, quantizer master immutability on both attention paths.");
    }

    private static void CheckAttentionParity(WeightSet initial, bool cuda)
    {
        var device = cuda ? torch.CUDA : torch.CPU;
        foreach (int length in new[] {1,7,31})
        {
            using var scope = torch.NewDisposeScope();
            using var explicitModel = new TorchModel(initial,device,useSdpa:false);
            using var sdpaModel = new TorchModel(initial,device,useSdpa:true);
            var data = Enumerable.Range(0,2*length).Select(i=>(long)(6+i%200)).ToArray();
            var ids = torch.tensor(data,dtype:torch.ScalarType.Int64,device:device).reshape(2,length);
            var x = explicitModel.Forward(ids); var y = sdpaModel.Forward(ids);
            double difference = (x-y).abs().max().item<float>();
            if(difference>0.002)throw new Exception($"SDPA forward mismatch at {length}: {difference}");
            x.pow(2).mean().backward(); y.pow(2).mean().backward();
            for(int i=0;i<explicitModel.Parameters.Count;i++)
            {
                var a = explicitModel.Parameters[i].grad ?? throw new Exception("Missing explicit gradient");
                var b = sdpaModel.Parameters[i].grad ?? throw new Exception("Missing SDPA gradient");
                double error=(a-b).abs().max().item<float>();
                double magnitude=a.abs().max().item<float>();
                if(!double.IsFinite(error)||error>0.002+0.003*magnitude)throw new Exception($"SDPA gradient mismatch at tensor {i}: {error}");
            }
        }
    }
}
