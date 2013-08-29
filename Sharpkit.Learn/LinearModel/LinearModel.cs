// -----------------------------------------------------------------------
// <copyright file="LinearModel.cs" company="Sharpkit.Learn">
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

using MathNet.Numerics.LinearAlgebra.Double;

namespace Sharpkit.Learn.LinearModel
{
    using System;

    /// <summary>
    /// Linear model base class.
    /// </summary>
    public abstract class LinearModel
    {
        /// <summary>
        /// Initializes a new instance of the LinearModel class.
        /// </summary>
        /// <param name="fitIntercept">Whether to calculate the intercept for this model. If set
        /// to false, no intercept will be used in calculations
        /// (e.g. data is expected to be already centered).</param>
        protected LinearModel(bool fitIntercept)
        {
            FitIntercept = fitIntercept;
        }

        /// <summary>
        /// Gets or sets a value indicating whether to calculate the intercept for this model. If set
        /// to false, no intercept will be used in calculations
        /// (e.g. data is expected to be already centered).
        /// </summary>
        public bool FitIntercept { get; set; }

        /// <summary>
        /// shape (n_targets, n_features)
        /// Estimated coefficients for problem.
        /// </summary>
        public Matrix CoefMatrix { get; set; }

        public Vector Coef
        {
            get { return (Vector)CoefMatrix.Column(0); }
            set { CoefMatrix = (Matrix)value.ToColumnMatrix(); }
        }

        /// <summary>
        /// Independent term in the linear model.
        /// </summary>
        public Vector InterceptVector { get; set; }

        public double Intercept
        {
            get { return InterceptVector[0]; }
            set { InterceptVector = DenseVector.OfEnumerable(new[] { value }); }
        }

        protected internal void SetIntercept(Vector xMean, Vector yMean, Vector xStd)
        {
            if (this.FitIntercept)
            {
                this.CoefMatrix.DivColumnVector(xStd, this.CoefMatrix);
                this.InterceptVector = new DenseVector(yMean.Count);
                for (int i = 0; i < yMean.Count; i++)
                {
                    this.InterceptVector[i] = yMean[i] - xMean.DotProduct(this.CoefMatrix.Column(i));
                }
            }
            else
            {
                this.InterceptVector = new DenseVector(yMean.Count);
            }
        }

        internal class CenterDataResult
        {
            public Matrix X { get; set; }
            public Matrix Y { get; set; }
            public Vector xMean { get; set; }
            public Vector yMean { get; set; }
            public Vector xStd { get; set; }
        }

        /// <summary>
        /// Centers data to have mean zero along axis 0. This is here because
        /// nearly all linear models will want their data to be centered.
        /// If sample_weight is not None, then the weighted mean of X and y
        /// is zero, and not the mean itself
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="fitIntercept"></param>
        /// <param name="normalize"></param>
        /// <param name="sampleWeight"></param>
        internal CenterDataResult CenterData(
            Matrix x,
            Matrix y,
            bool fitIntercept,
            bool normalize = false,
            Vector sampleWeight = null)
        {
            Vector xMean;
            Vector yMean = new DenseVector(y.ColumnCount);
            Vector xStd;

            if (fitIntercept)
            {
                if (x is SparseMatrix)
                {
                    xMean = DenseVector.Create(x.ColumnCount, i => 0.0);
                    xStd = DenseVector.Create(x.ColumnCount, i => 1.0);
                }
                else
                {
                    if (sampleWeight == null)
                    {
                        xMean = x.MeanRows();
                    }
                    else
                    {
                        xMean = (Vector)x.MulColumnVector(sampleWeight).SumRows().Divide(sampleWeight.Sum());
                    }

                    x = x.SubtractRowVector(xMean);

                    if (normalize)
                    {
                        xStd = new DenseVector(x.ColumnCount);

                        foreach (var row in x.RowEnumerator())
                        {
                            xStd.Add(row.Item2.PointwiseMultiply(row.Item2), xStd);
                        }

                        xStd.MapInplace(Math.Sqrt);

                        for (int i = 0; i < xStd.Count; i++)
                        {
                            if (xStd[i] == 0)
                            {
                                xStd[i] = 1;
                            }
                        }

                        x.DivRowVector(xStd, x);
                    }
                    else
                    {
                        xStd = DenseVector.Create(x.ColumnCount, i => 1.0);
                    }
                }

                if (sampleWeight == null)
                {
                    yMean = y.MeanRows();
                }
                else
                {
                    yMean = (Vector)(y.MulColumnVector(sampleWeight).SumRows() / sampleWeight.Sum());
                }

                y = (Matrix)y.Clone();
                foreach (var r in y.RowEnumerator())
                {
                    y.SetRow(r.Item1, r.Item2.Subtract(yMean));
                }
            }
            else
            {
                xMean = DenseVector.Create(x.ColumnCount, i => 0);
                xStd = DenseVector.Create(x.ColumnCount, i => 1);
                //y_mean = 0;
            }

            return new CenterDataResult {X = x, Y = y, xMean = xMean, yMean = yMean, xStd = xStd};
        }
    }
}
