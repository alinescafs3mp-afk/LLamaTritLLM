using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

internal static class LearningPlanSelfTest
{
    public static void Run(bool requireCuda,string root)
    {
        var config=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,Context=64,GroupSize=8};
        var resources=new ResourceOptions{Threads=1,MemoryMiB=4096,BatchSize=2,SequenceLength=64,PreferCuda=requireCuda};
        var options=new TrainingOptions{Steps=20,LearningRate=.001,WarmupCosine=true};
        var data=Dataset.EncodeAll([Dataset.Make("q","a"),Dataset.Make("x","b")],64,1);
        using var session=new TrainingSession(WeightSet.Initialize(config),resources,options);
        if(requireCuda&&session.Model.Device.type!=DeviceType.CUDA)throw new Exception("Scheduled test fell back to CPU");
        for(int i=0;i<options.Steps;i++)
        {
            double actual=ManualLearningRate.At(options,i);session.TrainStep(data,actual,CancellationToken.None);
            if(session.LastLearningRate!=actual||session.Optimizer.ParamGroups.Any(g=>g.LearningRate!=actual))
                throw new Exception("Actual optimizer LR differs from scheduled update");
        }
        string optimizer=Path.Combine(root,"audit19-schedule-optimizer.bin");
        session.SaveOptimizer(optimizer);var state=session.State;
        using var loaded=new TrainingSession(session.CapturePublicationMaster(),resources,options);
        loaded.RestoreState(optimizer,state);
        if(loaded.LastLearningRate!=state.LastLearningRate||loaded.Step!=20)throw new Exception("Last used LR lost during resume");
        // A new explicitly requested job starts its own curve; it is NOT automatic continuation of old offsets.
        double next=ManualLearningRate.At(options,0);loaded.TrainStep(data,next,CancellationToken.None);
        if(loaded.LastLearningRate!=next||loaded.Step!=21)throw new Exception("New run used old schedule offset");
        Console.WriteLine("PASS actual scheduled optimizer rates, last-rate persistence and explicit new-run semantics");
    }
}
