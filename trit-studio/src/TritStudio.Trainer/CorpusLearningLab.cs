using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

// Opt-in measured learning experiment on the ACTUAL bundled corpus, never a user workspace.
// A successful exit means the experiment completed, NOT that the model speaks coherently.
public static class CorpusLearningLab
{
    public static int Run(string[] args)
    {
        string Value(string key, string fallback)
        {
            int i=Array.IndexOf(args,key);
            if(i<0)return fallback;
            if(i+1>=args.Length||args[i+1].StartsWith("--",StringComparison.Ordinal))throw new ArgumentException(key+" needs a value.");
            return args[i+1];
        }
        bool enforceGuard=args.Contains("--enforce-guard");
        int steps=int.Parse(Value("--steps","1000"),CultureInfo.InvariantCulture);
        int batch=int.Parse(Value("--batch","16"),CultureInfo.InvariantCulture);
        double lr=double.Parse(Value("--lr","0.001"),CultureInfo.InvariantCulture);
        string parent=Path.GetFullPath(Value("--output-dir",Path.Combine(Path.GetTempPath(),"trit-learning-lab")));
        ModelLibrary.NoLinks(parent);Directory.CreateDirectory(parent);
        string run=Path.Combine(parent,"run-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss",CultureInfo.InvariantCulture)+"-"+Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(run); // unique child; no previous experiment is overwritten
        var config=ModelConfig.Large;
        var resources=new ResourceOptions{Threads=Math.Min(4,Environment.ProcessorCount),MemoryMiB=16384,
            BatchSize=batch,SequenceLength=512,PreferCuda=args.Contains("--cuda")};
        var options=new TrainingOptions{Steps=steps,LearningRate=lr,WarmupCosine=args.Contains("--scheduled") || args.Contains("--course"),PublishEvery=100,ConversationCourse=args.Contains("--course"),ContextPractice=args.Contains("--context-practice"),TransferPractice=args.Contains("--transfer-practice"),EqualExampleWeight=args.Contains("--course")};
        options.Validate();resources.ValidateInitialization(config);
        if(steps is < 1 or > 100000)throw new ArgumentException("The isolated learning lab supports 1..100000 steps per experiment.");
        torch.set_num_interop_threads(1);
        string data=Path.Combine(AppContext.BaseDirectory,"data");
        var training=BundledCorpus.Load(data,"seed.jsonl");var controls=BundledCorpus.Load(data,"validation.jsonl");
        var guard=new ValidationGuard(controls,512);guard.EnsureTraining(training);
        TrainingExample[] starters=[]; TrainingExample[] contexts=[]; TrainingExample[] language22=[]; TrainingExample[] facts22=[];
        if(options.ConversationCourse)
        {
            starters=BundledCorpus.Load(data,"conversation-starter.jsonl").Where(x=>!guard.Contains(x)).ToArray();
            if(options.ContextPractice) contexts=BundledCorpus.Load(data,"conversation-context.jsonl").Where(x=>!guard.Contains(x)).ToArray();
            if(options.TransferPractice)
            {
                language22=BundledCorpus.Load(data,"conversation-language.jsonl").Where(x=>!guard.Contains(x)).ToArray();
                facts22=BundledCorpus.Load(data,"conversation-transfer.jsonl").Where(x=>!guard.Contains(x)).ToArray();
            }
            training=ConversationSupervision.Expand(training.Concat(starters).Concat(contexts).Concat(language22).Concat(facts22).DistinctBy(x=>x.Id).ToArray(),guard).Examples;
        }
        var encoded=Dataset.EncodeAll(training,512,resources.Threads);
        var curriculum=options.ConversationCourse && !options.TransferPractice ? new ConversationCurriculum(encoded,starters.Select(x=>x.Id),options.ContextPractice ? ConversationSupervision.Expand(contexts,guard).Examples.Select(x=>x.Id) : null) : null;
        var dialoguePlanner=options.TransferPractice ? new DialogueBatchPlanner(encoded,starters.Concat(language22),contexts.Concat(facts22)) : null;
        var control=Dataset.EncodeAll(controls,512,resources.Threads);
        var byteReference=new ByteBaseline(encoded).Evaluate(control);
        using var life=new CancellationTokenSource();
        ConsoleCancelEventHandler interrupt=(_,e)=>{e.Cancel=true;life.Cancel();};Console.CancelKeyPress+=interrupt;
        var watch=Stopwatch.StartNew();var observations=new List<object>();
        try
        {
            double? bestControl=null; int lastSavedStep=0;
            using var session=new TrainingSession(WeightSet.Initialize(config),resources,options);
            if(resources.PreferCuda&&session.Model.Device.type!=DeviceType.CUDA)
                throw new InvalidOperationException("Learning lab requested CUDA but received CPU.");
            for(int i=0;i<=steps;i++)
            {
                life.Token.ThrowIfCancellationRequested();
                if(i>0)session.TrainStep(dialoguePlanner is null ? curriculum?.At(i-1,steps).Examples ?? encoded : encoded,ManualLearningRate.At(options,i-1),life.Token,equalExampleWeight:options.EqualExampleWeight,dialoguePlanner:dialoguePlanner,courseCompleted:i-1,courseTotal:steps);
                if(i!=0&&i%(options.ConversationCourse?500:100)!=0&&i!=steps)continue;
                double loss=session.Evaluate(control,life.Token);
                bool guardWouldReject=bestControl is double anchor && loss > anchor+Math.Max(.02,anchor*options.MaxValidationRegression);
                if(enforceGuard && guardWouldReject)
                {
                    observations.Add(new{step=i,controlLoss=loss,guardWouldReject,lastSavedStep,candidateExported=false});
                    JsonData.AtomicWrite(Path.Combine(run,"experiment.json"),new{scope="Real native guarded experiment stopped before exporting rejected candidate; NOT a fluency pass",
                        version=BundledCorpus.Version,config,resources,options,enforceGuard,lastSavedStep,stopReason="validation_regression",byteReference,observations});
                    Console.Error.WriteLine($"Learning lab rejected step {i}; retained prior exported step {lastSavedStep}. Report: {run}");
                    return 3;
                }
                bestControl=Math.Min(bestControl ?? loss,loss);
                string modelFile=Path.Combine(run,$"step-{i:D6}.tritmodel");
                ModelFiles.Write(modelFile,session.CapturePublicationMaster(life.Token),true,ct:life.Token);
                var cpu=new ManagedInference(ModelFiles.Read(modelFile,true,ct:life.Token),i,resources.Threads);
                var replies=new[]{"Привет!","Как дела?","Как тебя зовут?"}.Select(p=>
                {
                    string answer=cpu.Generate(ByteTokenizer.Prompt([],p,config.Context,128),()=>new SamplingOptions(0,262,1,1,128),null,life.Token);
                    return new{prompt=p,answer,health=GenerationHealth.Inspect(answer)};
                }).ToArray();
                lastSavedStep=i;
                observations.Add(new{step=i,guardWouldReject,seconds=watch.Elapsed.TotalSeconds,actualLearningRate=session.LastLearningRate,
                    controlLoss=loss,controlAccuracy=session.LastEvaluationAccuracy,replies,modelFile,
                    coursePhase=i==0?"не обучена":dialoguePlanner is not null ? DialogueBatchPlanner.PhaseName(i-1,steps) : curriculum?.At(i-1,steps).Name,
                    conversation=options.ConversationCourse ? ConversationProbe.Run(cpu,session.Step,ModelFiles.Hash(modelFile),life.Token) : null,
                    transfer=options.TransferPractice && i==steps ? GeneralizationProbe.Run(cpu,session.Step,ModelFiles.Hash(modelFile),BundledCorpus.Load(data,"transfer-challenge.jsonl"),training,ct:life.Token) : null,
                    contextTransfer=options.ContextPractice && (i==steps) ? ContextTransferProbe.Run(cpu,session.Step,ModelFiles.Hash(modelFile),BundledCorpus.Load(data,"context-challenge.jsonl"),16,life.Token) : null});
                JsonData.AtomicWrite(Path.Combine(run,"experiment.json"),new{scope="Real native corpus training and saved-file CPU generation; NOT a fluency certification",
                    version=BundledCorpus.Version,enforceGuard,contextSha256=options.ContextPractice ? ModelFiles.Hash(Path.Combine(data,"conversation-context.jsonl")) : null,seedSha256=ModelFiles.Hash(Path.Combine(data,"seed.jsonl")),
                    controlSha256=ModelFiles.Hash(Path.Combine(data,"validation.jsonl")),config,resources,options,device=session.DeviceName,
                    byteReference,observations});
                Console.Error.WriteLine($"Learning lab: step {i}/{steps}, loss {loss:F4}, control {session.LastEvaluationAccuracy?.Percent:F2}%, reference {100*byteReference.Accuracy:F2}%, report {run}");
            }
            Console.WriteLine(Path.Combine(run,"experiment.json"));return 0;
        }
        finally{Console.CancelKeyPress-=interrupt;}
    }
}
