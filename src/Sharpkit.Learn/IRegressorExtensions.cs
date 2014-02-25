// -----------------------------------------------------------------------
// <copyright file="IRegressorExtensions.cs" company="Sharpkit.Learn">
//  Copyright Sergey Zyuzin 2014.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// <see cref="IRegressor"/> extension methods.
    /// </summary>
    public static class IRegressorExtensions
    {
        /// <summary>
        /// Predict target values for sample x.
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">Array of length [nFeatures].</param>
        /// <returns>Returns predicted values. Array of length [nTargets].</returns>
        public static double[] Predict(this IRegressor regressor, double[] x)
        {
            return regressor.Predict(x.ToDenseVector().ToRowMatrix()).Row(0).ToArray();
        }

        /// <summary>
        /// Predict target values for samples in x.
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">Array with dimensions [nSamples, nFeatures].</param>
        /// <returns>Returns predicted values. Array with dimensions [nSamples, nTargets].</returns>
        public static double[,] Predict(this IRegressor regressor, double[,] x)
        {
            return regressor.Predict(DenseMatrix.OfArray(x)).ToArray();
        }

        /// <summary>
        /// Predict target values for samples in x.
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">Array with dimensions [nSamples, nFeatures].</param>
        /// <returns>Returns predicted values. Array with dimensions [nSamples].</returns>
        public static double[] PredictSingle(this IRegressor regressor, double[,] x)
        {
            return regressor.Predict(DenseMatrix.OfArray(x)).Column(0).ToArray();
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">
        /// Array with dimensions [nSamples, nFeatures].
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">Array with dimensions [nSamples]. Target values.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Array with dimensions [nSamples].</param>
        public static void Fit(this IRegressor regressor, double[,] x, double[] y, double[] sampleWeight = null)
        {
            Fit(regressor, x.ToDenseMatrix(), y.ToDenseVector(), sampleWeight.ToDenseVector());
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">
        /// Array with dimensions [nSamples, nFeatures].
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">Array with dimensions [nSamples, nTargets]. Target values.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Array with dimensions [nSamples].</param>
        public static void Fit(this IRegressor regressor, double[,] x, double[,] y, double[] sampleWeight = null)
        {
            regressor.Fit(x.ToDenseMatrix(), y.ToDenseMatrix(), sampleWeight.ToDenseVector());
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">
        /// Matrix with dimensions [nSamples, nFeatures].
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">Vector with dimensions [nSamples]. Target values.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Vector with dimensions [nSamples].</param>
        public static void Fit(
            this IRegressor regressor,
            Matrix<double> x,
            Vector<double> y,
            Vector<double> sampleWeight = null)
        {
            regressor.Fit(x, y.ToColumnMatrix(), sampleWeight);
        }

        /// <summary>
        /// <para>
        /// Returns the coefficient of determination R^2 of the prediction.
        /// </para>
        /// <para>
        /// The coefficient R^2 is defined as (1 - u/v), where u is the regression
        /// sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        /// sum of squares ((y_true - y_true.mean()) ** 2).sum().
        /// Best possible score is 1.0, lower values are worse.
        /// </para>
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">Feature matrix. Matrix with dimensions [nSamples, nFeatures].</param>
        /// <param name="y">True values for <paramref name="x"/>. Matrix with dimensions [nSamples, nTargets].</param>
        /// <returns> R^2 of Predict(x) wrt. y.</returns>
        public static double Score(this IRegressor regressor, Matrix<double> x, Matrix<double> y)
        {
            return Metrics.Metrics.R2Score(y, regressor.Predict(x));
        }

        /// <summary>
        /// <para>
        /// Returns the coefficient of determination R^2 of the prediction.
        /// </para>
        /// <para>
        /// The coefficient R^2 is defined as (1 - u/v), where u is the regression
        /// sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        /// sum of squares ((y_true - y_true.mean()) ** 2).sum().
        /// Best possible score is 1.0, lower values are worse.
        /// </para>
        /// </summary>
        /// <param name="regressor">A <see cref="IRegressor"/> instance.</param>
        /// <param name="x">Feature matrix. Matrix with dimensions [nSamples, nFeatures].</param>
        /// <param name="y">True values for <paramref name="x"/>. Vector with dimensions [nSamples].</param>
        /// <returns> R^2 of Predict(X) wrt. y.</returns>
        public static double Score(this IRegressor regressor, Matrix<double> x, Vector<double> y)
        {
            return Metrics.Metrics.R2Score(y.ToColumnMatrix(), regressor.Predict(x));
        }
    }
}
