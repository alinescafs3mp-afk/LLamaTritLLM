using TritStudio.Core;
internal static class LearningProgressChecks
{
    public static (string, Action)[] All =>
    [
        ("accuracy window weights tokens not equally sized batches", () => {
            var w=new AccuracyWindow(); w.Add(1,1,1,512); w.Add(0,9,2,512);
            Check(w.Current!.Correct==1&&w.Current.Total==10&&w.Current.Percent==10&&w.Current.Batches==2);
        }),
        ("accuracy window evicts only oldest batch and can decline honestly", () => {
            var w=new AccuracyWindow();for(int i=1;i<=32;i++)w.Add(2,2,i,512);
            w.Add(0,2,33,512); Check(w.Current!.Correct==62&&w.Current.Total==64&&w.Current.Percent<100&&w.Current.Batches==32);
        }),
        ("accuracy has no value until measured and reset clears model history", () => {
            var w=new AccuracyWindow();Check(w.Current is null);w.Add(0,3,1,64);Check(w.Current!.Percent==0);w.Clear();Check(w.Current is null);
        }),
        ("accuracy count validation rejects padding-like empty denominator and forged counts", () => {
            foreach(var a in new[]{new TokenAccuracy(0,0,0,512),new TokenAccuracy(-1,1,0,512),new TokenAccuracy(2,1,0,512),
                new TokenAccuracy(0,1,2,512),new TokenAccuracy(0,1,0,0),new TokenAccuracy(0,1,0,2049),new TokenAccuracy(0,1,0,512,0)})
                Throws<InvalidDataException>(()=>a.Validate(1));
        }),
        ("accuracy window rejects duplicate steps without mutation", () => {
            var w=new AccuracyWindow();w.Add(1,2,5,512);var a=w.Current;Throws<ArgumentException>(()=>w.Add(0,3,5,512));Check(w.Current==a);
        }),
        ("accuracy overflow refuses without poisoning prior totals", () => {
            var w=new AccuracyWindow();w.Add(0,long.MaxValue,1,512);var a=w.Current;
            Throws<OverflowException>(()=>w.Add(0,1,2,512));Check(w.Current==a);
        }),
        ("progress distinguishes zero accuracy from not measured", () => {
            var p=new LearningProgress(null,new TokenAccuracy(0,2,0,64));p.Validate(0);Check(p.Training is null&&p.Validation!.Percent==0);
        }),
        ("training accuracy cannot claim a zero-step model", () => Throws<InvalidDataException>(()=>new LearningProgress(new TokenAccuracy(1,1,0,64)).Validate(0))),
        ("progress checks model context and future observations", () => {
            Throws<InvalidDataException>(()=>new LearningProgress(new TokenAccuracy(1,1,4,512)).Validate(3));
            Throws<InvalidDataException>(()=>new LearningProgress(null,new TokenAccuracy(1,1,3,512)).Validate(3,128));
        }),
        ("progress storage is per checkpoint and old records remain optional", () => {
            var p=new LearningProgress(new TokenAccuracy(3,4,9,512,2),new TokenAccuracy(7,10,9,512,1));
            var json=System.Text.Json.JsonSerializer.Serialize(p,JsonData.Options);
            var copy=System.Text.Json.JsonSerializer.Deserialize<LearningProgress>(json,JsonData.Options);Check(copy==p);
            Check(new StatusEvent("legacy").Accuracy is null);
        }),
        ("progress window supports true batch64 with full target counts", () => {
            var examples=Enumerable.Range(0,64).Select(i=>Dataset.Encode(Dataset.Make("пример "+i,"да"),64)).ToArray();
            var b=SupervisedBatch.Build(examples,64);var w=new AccuracyWindow();w.Add(b.Targets.Length/2,b.Targets.Length,1,64);
            Check(w.Current!.Total==examples.Sum(x=>x.Labels.Count(y=>y!=-100)));
        }),
    ];
    private static void Check(bool c){if(!c)throw new Exception("Learning progress contract failed.");}
    private static void Throws<T>(Action a)where T:Exception{try{a();}catch(T){return;}throw new Exception("Expected "+typeof(T).Name);}
}
