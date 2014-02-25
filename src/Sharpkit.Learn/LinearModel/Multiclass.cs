// -----------------------------------------------------------------------
// <copyright file="Multiclass.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    /// <summary>
    /// Determines how to handle multiclass problem.
    /// </summary>
    public enum Multiclass
    {
        /// <summary>
        /// Trains n_classes one-vs-rest classifiers.
        /// </summary>
        Ovr,

        /// <summary>
        /// Optimizes a joint objective over all classes.
        /// While `crammer_singer` is interesting from an theoretical perspective
        /// as it is consistent it is seldom used in practice and rarely leads to
        /// better accuracy and is more expensive to compute.
        /// </summary>
        CrammerSinger
    }
}