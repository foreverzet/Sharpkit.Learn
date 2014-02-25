// -----------------------------------------------------------------------
// <copyright file="PrecisionRecallResultAvg.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Metrics
{
    /// <summary>
    /// Return value of <see cref="Metrics.PrecisionScoreAvg"/>.
    /// </summary>
    public struct PrecisionRecallResultAvg
    {
        /// <summary>
        /// Gets or sets precision.
        /// </summary>
        public double Precision { get; set; }

        /// <summary>
        /// Gets or sets recall.
        /// </summary>
        public double Recall { get; set; }

        /// <summary>
        /// Gets or sets fbeta score.
        /// </summary>
        public double FBetaScore { get; set; }
    }
}