// -----------------------------------------------------------------------
// <copyright file="ILinearModel.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class LinearModel
    {
        /// <summary>
        /// shape (n_targets, n_features)
        /// Estimated coefficients for problem.
        /// </summary>
        public Matrix<double> Coef { get; set; }

        /// <summary>
        /// Independent term in the linear model.
        /// </summary>
        public Vector<double> Intercept { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to calculate the intercept for this model. If set
        /// to false, no intercept will be used in calculations
        /// (e.g. data is expected to be already centered).
        /// </summary>
        public bool FitIntercept { get; set; }

        internal LinearModel(bool fitIntercept)
        {
            FitIntercept = fitIntercept;
        }

        public Matrix<double> DecisionFunction(Matrix<double> x)
        {
            int nFeatures = this.Coef.ColumnCount;
            if (x.ColumnCount != nFeatures)
            {
                throw new ArgumentException(
                    string.Format(
                    "X has {0} features per sample; expecting {1}",
                    x.ColumnCount,
                    nFeatures));
            }


            // todo: use TransposeAndMultiply. But there's bug in Math.Net
            // which appears with sparse matrices.
            var tmp = x.Multiply(this.Coef.Transpose());
            tmp.AddRowVector(this.Intercept, tmp);
            //tmp.MapIndexedInplace((i, j, v) => v + this.InterceptVector[j]);
            return tmp;
        }

        internal void SetIntercept(Vector<double> xMean, Vector<double> yMean, Vector<double> xStd)
        {
            if (FitIntercept)
            {
                this.Coef.DivRowVector(xStd, this.Coef);

                this.Intercept = (xMean.ToRowMatrix().TransposeAndMultiply(this.Coef) * (-1)).Row(0) + yMean;
            }
            else
            {
                this.Intercept = new DenseVector(yMean.Count);
            }
        }

        internal class CenterDataResult
        {
            public Matrix<double> X { get; set; }
            public Matrix<double> Y { get; set; }
            public Vector<double> xMean { get; set; }
            public Vector<double> yMean { get; set; }
            public Vector<double> xStd { get; set; }
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
        internal static CenterDataResult CenterData(
            Matrix<double> x,
            Matrix<double> y,
            bool fitIntercept,
            bool normalize = false,
            Vector<double> sampleWeight = null)
        {
            Vector<double> xMean;
            Vector<double> yMean = new DenseVector(y.ColumnCount);
            Vector<double> xStd;

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
                        xMean = x.MeanOfEveryColumn();
                    }
                    else
                    {
                        xMean = x.MulColumnVector(sampleWeight).SumOfEveryColumn().Divide(sampleWeight.Sum());
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
                    yMean = y.MeanOfEveryColumn();
                }
                else
                {
                    yMean = y.MulColumnVector(sampleWeight).SumOfEveryColumn() / sampleWeight.Sum();
                }

                y = y.Clone();
                y = y.SubtractRowVector(yMean);
            }
            else
            {
                xMean = DenseVector.Create(x.ColumnCount, i => 0);
                xStd = DenseVector.Create(x.ColumnCount, i => 1);
            }

            return new CenterDataResult { X = x, Y = y, xMean = xMean, yMean = yMean, xStd = xStd };
        }

        /// <summary>
        /// Probability estimation for OvR logistic regression.
        ///
        /// Positive class probabilities are computed as
        /// 1. / (1. + np.exp(-self.decision_function(X)));
        /// multiclass is handled by normalizing that over all classes.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal Matrix<double> PredictProbaLrInternal(Matrix<double> x)
        {
            var prob = this.DecisionFunction(x);
            prob.MapInplace(v => 1.0 / (Math.Exp(-v) + 1));

            if (prob.ColumnCount == 1)
            {
                var p1 = prob.Clone();
                p1.MapInplace(v => 1 - v);
                return p1.HStack(prob);
            }
            else
            {
                // OvR normalization, like LibLinear's predict_probability
                prob.DivColumnVector(prob.SumOfEveryRow(), prob);
                return prob;
            }
        }
    }
}
