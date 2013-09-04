// -----------------------------------------------------------------------
// <copyright file="LinearClassifier.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Base class for linear classifiers.
    /// Handles prediction for sparse and dense X.
    /// </summary>
    public abstract class LinearClassifier<TLabel>: LinearModel where TLabel:IEquatable<TLabel>
    {
        protected LinearClassifier(bool fitIntercept, ClassWeight<TLabel> classWeight) : base(fitIntercept)
        {
            ClassWeight = classWeight ?? ClassWeight<TLabel>.Uniform;
        }

        /// <summary>
        /// Predict confidence scores for samples.
        ///
        /// The confidence score for a sample is the signed distance of that
        /// sample to the hyperplane.
        /// </summary>
        /// <param name="x">shape = [n_samples, n_features] Samples.</param>
        /// <returns>[n_samples,n_classes]
        ///    Confidence scores per (sample, class) combination. In the binary
        ///    case, confidence score for the "positive" class.</returns>
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

        /// <summary>
        /// Probability estimation for OvR logistic regression.
        ///
        /// Positive class probabilities are computed as
        /// 1. / (1. + np.exp(-self.decision_function(X)));
        /// multiclass is handled by normalizing that over all classes.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public Matrix<double> PredictProbaLr(Matrix<double> x)
        {
            var prob = this.DecisionFunction(x);
            prob.MapInplace( v => 1.0 / (Math.Exp(-v) + 1));

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

        /// <summary>
        /// Predict class labels for samples in X.
        /// </summary>
        /// <param name="x">[n_samples, n_features] Samples.</param>
        /// <returns>[n_samples] Predicted class label per sample.</returns>
        public TLabel Predict(double[] x)
        {
            return this.Predict(x.ToDenseVector().ToRowMatrix())[0];
        }

        /// <summary>
        /// Predict class labels for samples in X.
        /// </summary>
        /// <param name="x">[n_samples, n_features] Samples.</param>
        /// <returns>[n_samples] Predicted class label per sample.</returns>
        public TLabel[] Predict(Matrix<double> x)
        {
            var scores = this.DecisionFunction(x);
            if (scores.ColumnCount == 1)
            {
                return scores.Column(0).Select(v => v > 0 ? Classes[1] : Classes[0]).ToArray();
            }
            else
            {
                return scores.RowEnumerator().Select(r => Classes[r.Item2.MaximumIndex()]).ToArray();
            }
        }

        public abstract TLabel[] Classes { get; }
        
        protected readonly ClassWeight<TLabel> ClassWeight;
    }
}
