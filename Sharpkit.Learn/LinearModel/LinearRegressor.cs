// -----------------------------------------------------------------------
// <copyright file="LinearModel.cs" company="Sharpkit.Learn">
// Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
//         Fabian Pedregosa <fabian.pedregosa@inria.fr>
//         Olivier Grisel <olivier.grisel@ensta.org>
//         Vincent Michel <vincent.michel@inria.fr>
//         Peter Prettenhofer <peter.prettenhofer@gmail.com>
//         Mathieu Blondel <mathieu@mblondel.org>
//         Lars Buitinck <L.J.Buitinck@uva.nl>
//         Sergey Zyuzin
//
// License: BSD 3 clause
// </copyright>

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Generalized Linear models.
    /// </summary>
    public abstract class LinearRegressor : LinearModel
    {
        /// <summary>
        /// Initializes a new instance of the LinearRegressor class.
        /// </summary>
        /// <param name="fitIntercept"></param>
        protected LinearRegressor(bool fitIntercept) : base(fitIntercept)
        {
        }

        public LinearRegressor Fit(Matrix<double> x, Vector<double> y, Vector<double> sampleWeight = null)
        {
            return Fit(x, y.ToColumnMatrix(), sampleWeight);
        }

        /// <summary>
        /// Fit linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples,n_features]. Training data</param>
        /// <param name="y">Target values.[n_samples, n_targets]</param>
        /// <param name="sampleWeight">Sample weights.[n_samples]</param>
        /// <returns>Instance of self.</returns>
        public abstract LinearRegressor Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null);

        /// <summary>
        /// Decision function of the linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples, n_features].</param>
        /// <returns>shape = [n_samples] Returns predicted values.</returns>
        public Matrix<double> DecisionFunction(Matrix<double> x)
        {
            var tmp = x.TransposeAndMultiply(this.Coef);
            tmp.MapIndexedInplace((i, j,v) => v + this.Intercept[j]);
            return tmp;
        }
 
        /// <summary>
        /// Predict using the linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples, n_features].</param>
        /// <returns>shape = [n_samples] Returns predicted values.</returns>
        public Matrix<double> Predict(Matrix<double> x)
        {
            return this.DecisionFunction(x);
        }
    }
}
