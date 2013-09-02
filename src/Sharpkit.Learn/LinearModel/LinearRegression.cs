// -----------------------------------------------------------------------
// <copyright file="LinearRegression.cs" company="Sharpkit.Learn">
// Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
//         Fabian Pedregosa <fabian.pedregosa@inria.fr>
//         Olivier Grisel <olivier.grisel@ensta.org>
//         Vincent Michel <vincent.michel@inria.fr>
//         Peter Prettenhofer <peter.prettenhofer@gmail.com>
//        Mathieu Blondel <mathieu@mblondel.org>
//         Lars Buitinck <L.J.Buitinck@uva.nl>
//
// License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Generic.Factorization;

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using System.Threading.Tasks;
    using MathNet.Numerics.LinearAlgebra.Double;
    using LeastSquares;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Ordinary least squares Linear Regression.
    /// </summary>
    public class LinearRegression : LinearRegressor
    {
        /// <summary>
        /// Gets or sets a value indicating whether regressors X shall be normalized
        /// before regression.
        /// </summary>
        private bool Normalize { get; set; }

        /// <summary>
        /// Initializes a new instance of the LinearRegression class.
        /// </summary>
        /// <param name="fitIntercept">Whether to calculate the intercept for this model. If set
        /// to false, no intercept will be used in calculations
        /// (e.g. data is expected to be already centered).</param>
        /// <param name="normalize">If True, the regressors X will be normalized before regression.</param>
        public LinearRegression(bool fitIntercept = true, bool normalize = false) : base(fitIntercept)
        {
            this.Normalize = normalize;
        }

        /// <summary>
        /// Fit linear model.
        /// </summary>
        /// <param name="x">Matrix of shape [n_samples,n_features]. Training data</param>
        /// <param name="y">Target values.[n_samples, n_targets]</param>
        /// <param name="sampleWeight">Sample weights.[n_samples]</param>
        /// <returns>Instance of self.</returns>
        public override LinearRegressor Fit(Matrix<double> x, Matrix<double> y, Vector<double> sampleWeight = null)
        {
            var centerDataResult = CenterData(x, y, this.FitIntercept, this.Normalize);
            x = centerDataResult.X;
            y = centerDataResult.Y;

            if (x is SparseMatrix)
            {
                this.Coef = new DenseMatrix(y.ColumnCount, x.ColumnCount);
                Parallel.ForEach(y.ColumnEnumerator(), c=>
                    this.Coef.SetRow(c.Item1, Lsqr.lsqr(x, c.Item2).X));
            }
            else
            {
                this.Coef = x.SvdSolve(y).Transpose();
            }

            this.SetIntercept(centerDataResult.xMean, centerDataResult.yMean, centerDataResult.xStd);
            return this;
        }
    }
}
