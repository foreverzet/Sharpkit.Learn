// -----------------------------------------------------------------------
// <copyright file="Metrics.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Metrics
{
    /// <summary>
    /// Resource of calculating precision/recall.
    /// </summary>
    public struct PrecisionRecallResult
    {
        /// <summary>
        /// Gets or sets precision value.
        /// </summary>
        public double[] Precision { get; set; }

        /// <summary>
        /// Gets or sets recall value.
        /// </summary>
        public double[] Recall { get; set; }

        /// <summary>
        /// Gets or sets FBetaScore value.
        /// </summary>
        public double[] FBetaScore { get; set; }

        /// <summary>
        /// Gets or sets support value.
        /// </summary>
        public int[] Support { get; set; }
    }
}