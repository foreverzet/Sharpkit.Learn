// -----------------------------------------------------------------------
// <copyright file="AverageKind.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Metrics
{
    using System;

    /// <summary>
    /// Type of averaging performed on the data.
    /// </summary>
    public enum AverageKind
    {
        /// <summary>
        /// Calculate metrics globally by counting the total true positives,
        /// false negatives and false positives.
        /// </summary>
        Micro,

        /// <summary>
        /// Calculate metrics for each label, and find their unweighted
        /// mean. This does not take label imbalance into account.
        /// </summary>
        Macro,

        /// <summary>
        /// Calculate metrics for each label, and find their average, weighted
        /// by support (the number of true instances for each label). This
        /// alters 'macro' to account for label imbalance; it can result in an
        /// F-score that is not between precision and recall.
        /// </summary>
        Weighted
    }
}
