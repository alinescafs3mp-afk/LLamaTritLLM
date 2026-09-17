using TorchSharp;
using TritStudio.Core;
namespace TritStudio.Trainer;

// Real optimizer and owned RNG must consume the SAME planned batch; no second resampling.
public static class DialogueBatchSelfTest
{
    public static void Run(bool requireCuda,string root)
    {
        var config=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=512,Seed=42};
        var language=new[]{Dataset.Make("Приветик.","Привет! Как день?"),Dataset.Make("Книга закончилась.","Как тебе конец?")};
        var facts=new[]{Dataset.Make("Как зовут?","Оля.",history:[new("Я Оля.","Понял.",0)]),Dataset.Make("Как зовут?","Рома.",history:[new("Я Рома.","Понял.",0)])};
        var rows=language.Concat(facts).Append(Dataset.Make("Текст для общего корпуса.")).ToArray();
        var encoded=Dataset.EncodeAll(rows,256,1);
        foreach(int b in new[]{8,64})
        {
            var resources=new ResourceOptions{Threads=1,MemoryMiB=4096,SequenceLength=256,BatchSize=b,PreferCuda=requireCuda};
            var options=new TrainingOptions{ConversationCourse=true,ContextPractice=true,TransferPractice=true,EqualExampleWeight=true};
            using var session=new TrainingSession(WeightSet.Initialize(config),resources,options);
            if(requireCuda && session.Model.Device.type!=DeviceType.CUDA)throw new Exception("v22 native test got CPU instead of requested CUDA");
            var planner=new DialogueBatchPlanner(encoded,language,facts);var reference=new DialogueBatchPlanner(encoded,language,facts);
            var rng=new SamplerRandom((ulong)config.Seed);var expected=reference.Select(b,rng,0,100);
            var before=session.CapturePublicationMaster();
            var result=session.TrainStep(encoded,.001,CancellationToken.None,equalExampleWeight:true,dialoguePlanner:planner,courseCompleted:0,courseTotal:100);
            if(session.State.SamplerState!=rng.State||result.TargetTokens!=expected.TargetTokens||session.LastStepPerformance?.DialogueMix!=reference.LastMix)
                throw new Exception("Actual v22 train path resampled or misreported its planned batch");
            if(session.Step!=1||before.CountChanged(session.CapturePublicationMaster())==0)throw new Exception("v22 planned batch did not update weights");
            double val=session.Evaluate(encoded);if(!double.IsFinite(val))throw new Exception("v22 training corrupted standard evaluation");
            string optimizer=Path.Combine(root,$"v22-optimizer-{b}.bin");session.SaveOptimizer(optimizer);
            var state=session.State;var master=session.CapturePublicationMaster();
            using var resumed=new TrainingSession(master,resources,options);resumed.RestoreState(optimizer,state);
            var again=new DialogueBatchPlanner(encoded,language,facts);
            session.TrainStep(encoded,.001,CancellationToken.None,equalExampleWeight:true,dialoguePlanner:planner,courseCompleted:1,courseTotal:100);
            resumed.TrainStep(encoded,.001,CancellationToken.None,equalExampleWeight:true,dialoguePlanner:again,courseCompleted:1,courseTotal:100);
            if(session.State!=resumed.State)throw new Exception("v22 optimizer/RNG state differs after resume");
            var a=session.CapturePublicationMaster();var c=resumed.CapturePublicationMaster();
            foreach(var name in a.Values.Keys)
            for(int i=0;i<a.Values[name].Length;i++)
                if(Math.Abs(a.Values[name][i]-c.Values[name][i])>0.0002f)throw new Exception("v22 restored native weights diverged");
        }
        Console.WriteLine("PASS native v22 planned batch8/64, exact RNG consumption, real update, optimizer resume and ordinary control evaluation");
    }
}
