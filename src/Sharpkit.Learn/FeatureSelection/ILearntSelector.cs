// -----------------------------------------------------------------------
// <copyright file="ILearntSelector.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.FeatureSelection
{
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Transformer mixin selecting features based on importance weights.
    /// Implemented by any estimator that can evaluate the relative
    /// importance of individual features for feature selection.
    /// </summary>
    public interface ILearntSelector
    {
        /// <summary>
        /// Importance weights.
        /// </summary>
        /// <returns>Importance weights.</returns>
        Vector<double> FeatureImportances();
    }
}
