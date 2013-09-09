// -----------------------------------------------------------------------
// <copyright file="SvcLoss.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Svm
{
    using System;

    /// <summary>
    /// Loss function.
    /// </summary>
    public enum SvcLoss
    {
        /// <summary>
        /// Hinge loss (standard SVM).
        /// </summary>
        L1,

        /// <summary>
        /// Squared hinge loss.
        /// </summary>
        L2
    }
}
