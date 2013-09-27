namespace Sharpkit.Learn.Metrics
{
    public struct PrecisionRecallResult
    {
        public double[] Precision { get; set; }

        public double[] Recall { get; set; }

        public double[] FBetaScore { get; set; }

        public int[] Support { get; set; }
    }
}