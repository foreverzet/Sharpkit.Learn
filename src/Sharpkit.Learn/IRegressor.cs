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
        void Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null);
        
        Matrix<double> Predict(Matrix<double> x);
    }
}
