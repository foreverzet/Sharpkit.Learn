// -----------------------------------------------------------------------
// <copyright file="IRegressorExtensions.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Double;

namespace Sharpkit.Learn
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public static class IRegressorExtensions
    {
        /// <summary>
        /// Predict using the linear model.
        /// </summary>
        /// <param name="x">Array of length [nFeatures].</param>
        /// <returns>Returns predicted value.</returns>
        public static double[] Predict(this IRegressor regressor, double[] x)
        {
            return regressor.Predict(x.ToDenseVector().ToRowMatrix()).Row(0).ToArray();
        }

        public static double PredictSingle(this IRegressor regressor, double[] x)
        {
            return regressor.Predict(x.ToDenseVector().ToRowMatrix())[0,0];
        }

        public static double[,] Predict(this IRegressor regressor, double[,] x)
        {
            return regressor.Predict(DenseMatrix.OfArray(x)).ToArray();
        }

        public static double[] PredictSingle(this IRegressor regressor, double[,] x)
        {
            return regressor.Predict(DenseMatrix.OfArray(x)).Column(0).ToArray();
        }

        public static void Fit(this IRegressor regressor, double[,] x, double[] y, double[] sampleWeight = null)
        {
            Fit(regressor, x.ToDenseMatrix(), y.ToDenseVector(), sampleWeight.ToDenseVector());
        }

        public static void Fit(this IRegressor regressor, double[,] x, double[,] y, double[] sampleWeight = null)
        {
            regressor.Fit(x.ToDenseMatrix(), y.ToDenseMatrix(), sampleWeight.ToDenseVector());
        }

        public static void Fit(this IRegressor regressor, Matrix<double> x, Vector<double> y, Vector<double> sampleWeight = null)
        {
            regressor.Fit(x, y.ToColumnMatrix(), sampleWeight);
        }

        public static double Score(this IRegressor regressor, Matrix<double> X, Matrix<double> y)
        {
        /*Returns the coefficient of determination R^2 of the prediction.


        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0, lower values are worse.


        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.


        y : array-like, shape = (n_samples,)
            True values for X.


        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        */


            return Metrics.Metrics.R2Score(y, regressor.Predict(X));
        }

        public static double Score(this IRegressor regressor, Matrix<double> x, Vector<double> y)
        {
            return Metrics.Metrics.R2Score(y.ToColumnMatrix(), regressor.Predict(x));
        }
    }
}
