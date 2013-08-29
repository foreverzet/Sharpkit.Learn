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
    using MathNet.Numerics.LinearAlgebra.Double;

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

        public LinearRegressor Fit(Matrix x, Vector y, Vector sampleWeight = null)
        {
            return Fit(x, (Matrix)y.ToColumnMatrix(), sampleWeight);
        }

        /// <summary>
        /// Fit linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples,n_features]. Training data</param>
        /// <param name="y">Target values.[n_samples, n_targets]</param>
        /// <param name="sampleWeight">Sample weights.[n_samples]</param>
        /// <returns>Instance of self.</returns>
        public abstract LinearRegressor Fit(Matrix x, Matrix y, Vector sampleWeight = null);

        /// <summary>
        /// Decision function of the linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples, n_features].</param>
        /// <returns>shape = [n_samples] Returns predicted values.</returns>
        public Matrix DecisionFunction(Matrix x)
        {
            var tmp = x.Multiply(this.CoefMatrix);
            tmp.MapIndexedInplace((i, j,v) => v + this.InterceptVector[j]);
            return (Matrix)tmp;
        }
 
        /// <summary>
        /// Predict using the linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples, n_features].</param>
        /// <returns>shape = [n_samples] Returns predicted values.</returns>
        public Matrix Predict(Matrix x)
        {
            return this.DecisionFunction(x);
        }
    }
}
