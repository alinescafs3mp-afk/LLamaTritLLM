using TorchSharp;
using TritStudio.Core;
using static TorchSharp.torch;
namespace TritStudio.Trainer;
public static class LearningProgressSelfTest
{
    public static void Run(bool requireCuda,string root)
    {
        var c=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=128};
        var e=Enumerable.Range(0,65).Select(i=>Dataset.Encode(Dataset.Make("номер "+i,i%2==0?"да":"нет"),64)).ToArray();
        foreach(bool selected in new[]{false,true})
        {
            var r=new ResourceOptions{Threads=1,BatchSize=64,SequenceLength=64,MemoryMiB=4096,PreferCuda=requireCuda,ProjectOnlyTargets=selected};
            using var session=new TrainingSession(WeightSet.Initialize(c),r,new TrainingOptions());
            if(requireCuda&&session.Model.Device.type!=DeviceType.CUDA)throw new Exception("Metrics test requested CUDA but got CPU.");
            Check(session.TrainingAccuracy is null,"untrained observation must be unknown");
            session.Evaluate(e); var measured=session.LastEvaluationAccuracy!;
            // Independent full-output teacher-forced oracle: count only non-ignored labels, not input/padding.
            long correct=0,total=0;
            using(var noGrad=no_grad())
            foreach(var x in e)
            {
                using var scope=NewDisposeScope();var ids=tensor(x.Tokens.Select(v=>(long)v).ToArray(),dtype:ScalarType.Int64,device:session.Model.Device).reshape(1,-1);
                var predicted=session.Model.Forward(ids).argmax(-1).cpu().data<long>().ToArray();
                for(int i=0;i<x.Labels.Length;i++)if(x.Labels[i]!=-100){total++;if(predicted[i]==x.Labels[i])correct++;}
            }
            Check(measured.Correct==correct&&measured.Total==total&&measured.Step==0&&measured.Batches==2,"full control oracle mismatch");
            long passes=session.ValidationPasses;session.Evaluate(e);Check(session.ValidationPasses==passes&&session.LastEvaluationAccuracy==measured,"cached metrics mismatch");
            session.TrainStep(e,.001,CancellationToken.None);var training=session.TrainingAccuracy!;
            Check(session.LastStepPerformance!.BatchSize==64&&training.Step==1&&training.Batches==1,"batch64 update/accuracy path mismatch");
            training.Validate(session.Step,c.Context);session.Evaluate(e);Check(session.LastEvaluationAccuracy!.Step==1&&session.ValidationPasses==passes+1,"new weights reused stale accuracy");
            var before=session.CapturePublicationMaster();string opt=Path.Combine(root,"metric-optimizer-"+selected+".bin");session.SaveOptimizer(opt);
            session.Restore(before,opt,session.State);Check(session.TrainingAccuracy is null&&session.LastEvaluationAccuracy is null,"restore kept unrelated live window");
            session.Evaluate(e);Check(session.LastEvaluationAccuracy!.Step==1,"restored control step mismatch");
        }
        Console.WriteLine("PASS actual native metrics: selected/full, batch64, all validation rows, ignored targets, real update, cache, restore.");
    }
    private static void Check(bool c,string m){if(!c)throw new Exception(m);}
}
