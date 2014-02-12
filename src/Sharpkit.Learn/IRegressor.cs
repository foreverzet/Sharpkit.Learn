// -----------------------------------------------------------------------
// <copyright file="IRegressor.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Interface implemented by all regressors.
    /// </summary>
    public interface IRegressor
    {
        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">
        /// Matrix with dimensions [nSamples, nFeatures].
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">Vector with dimensions [nSamples, nTargets]. Target values.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Array with dimensions [nSamples].</param>
        void Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null);

        /// <summary>
        /// Predict target values for samples in <paramref name="x"/>.
        /// </summary>
        /// <param name="x">Array with dimensions [nSamples, nFeatures].</param>
        /// <returns>Returns predicted values. Array with dimensions [nSamples, nTargets].</returns>
        Matrix<double> Predict(Matrix<double> x);
    }
}
