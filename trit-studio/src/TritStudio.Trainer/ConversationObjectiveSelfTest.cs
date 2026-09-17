using TorchSharp;
using TritStudio.Core;
using static TorchSharp.torch;
namespace TritStudio.Trainer;

// Real native oracle: equal-example CE must match a mean of per-row CE values and gradients.
// This tests the objective, not fluency or whether one training recipe is universally optimal.
public static class ConversationObjectiveSelfTest
{
    public static void Run(bool requireCuda, string root)
    {
        var config=new ModelConfig{Dimension=16,HiddenDimension=32,Layers=1,Heads=2,KvHeads=1,GroupSize=8,Context=128};
        var samples=Dataset.EncodeAll([Dataset.Make("a","да"),Dataset.Make("b","ответ длиннее"),
            Dataset.Make("c","нет",history:[new("раньше","контекст",0)])],96,1);
        var batch=SupervisedBatch.Build(samples,samples.Max(x=>x.Tokens.Length));
        var device=requireCuda?CUDA:CPU;
        foreach(bool compact in new[]{false,true})
        {
            using var scope=NewDisposeScope();
            var initial=WeightSet.Initialize(config);
            using var weighted=new TorchModel(initial,device);
            using var oracle=new TorchModel(initial,device);
            var ids=tensor(batch.Inputs,dtype:ScalarType.Int64,device:device).reshape(batch.Rows,batch.Length);
            var labels=tensor(batch.Targets,dtype:ScalarType.Int64,device:device);
            var pos=tensor(batch.Positions,dtype:ScalarType.Int64,device:device);
            var z=compact?weighted.Forward(ids,pos):weighted.Forward(ids).reshape(-1,262).index_select(0,pos);
            var logits=oracle.Forward(ids).reshape(-1,262).index_select(0,pos);
            var expectedParts=new List<Tensor>();
            for(int row=0;row<batch.Rows;row++)
            {
                long[] indices=Enumerable.Range(0,batch.Positions.Length).Where(i=>batch.Positions[i]/batch.Length==row).Select(i=>(long)i).ToArray();
                var index=tensor(indices,dtype:ScalarType.Int64,device:device);
                expectedParts.Add(nn.functional.cross_entropy(logits.index_select(0,index),labels.index_select(0,index)));
            }
            var expected=stack(expectedParts.ToArray()).mean();
            var actual=(nn.functional.cross_entropy(z,labels,reduction:nn.Reduction.None)*
                tensor(ExampleLossWeights.Build(batch),dtype:ScalarType.Float32,device:device)).sum();
            if(Math.Abs(actual.item<float>()-expected.item<float>())>.002)throw new Exception("Balanced objective value differs from per-example oracle.");
            actual.backward();expected.backward();
            for(int i=0;i<weighted.Parameters.Count;i++)
            {
                var a=weighted.Parameters[i].grad??throw new Exception("Missing balanced gradient");
                var b=oracle.Parameters[i].grad??throw new Exception("Missing oracle gradient");
                double error=(a-b).abs().max().item<float>();double scale=b.abs().max().item<float>();
                if(!double.IsFinite(error)||error>.002+.003*scale)throw new Exception("Balanced objective gradient mismatch.");
            }
        }
        var resources=new ResourceOptions{Threads=1,BatchSize=3,SequenceLength=96,MemoryMiB=4096,PreferCuda=requireCuda};
        using var session=new TrainingSession(WeightSet.Initialize(config),resources,new TrainingOptions());
        if(requireCuda&&session.Model.Device.type!=DeviceType.CUDA)throw new Exception("Conversation objective requested CUDA but got CPU.");
        var before=session.CapturePublicationMaster();session.TrainStep(samples,.001,CancellationToken.None,equalExampleWeight:true);
        if(session.Step!=1||before.CountChanged(session.CapturePublicationMaster())==0)throw new Exception("Balanced path did not update actual weights.");
        double control=session.Evaluate(samples);if(!double.IsFinite(control))throw new Exception("Balanced training corrupted ordinary control evaluation.");
        Console.WriteLine("PASS native conversation objective: unequal masked targets, selected/full, per-example loss+all gradients, real update and ordinary control evaluation.");
    }
}
