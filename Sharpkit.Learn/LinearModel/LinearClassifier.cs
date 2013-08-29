// -----------------------------------------------------------------------
// <copyright file="LinearClassifier.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// Base class for linear classifiers.
    /// Handles prediction for sparse and dense X.
    /// </summary>
    public abstract class LinearClassifier<TLabel>: LinearModel where TLabel:IEquatable<TLabel>
    {
        protected LinearClassifier(bool fitIntercept, ClassWeight<TLabel> classWeight) : base(fitIntercept)
        {
            class_weight = classWeight ?? ClassWeight<TLabel>.Uniform;
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
        public Matrix DecisionFunction(Matrix x)
        {
            int nFeatures = this.CoefMatrix.RowCount;
            if (x.ColumnCount != nFeatures)
            {
                throw new ArgumentException(
                    string.Format(
                    "X has {0} features per sample; expecting {1}",
                    x.ColumnCount,
                    nFeatures));
            }

            var tmp = x.Multiply(this.CoefMatrix);
            tmp.MapIndexedInplace((i, j, v) => v + this.InterceptVector[j]);
            return (Matrix)tmp;
        }

        /// <summary>
        /// Probability estimation for OvR logistic regression.
        ///
        /// Positive class probabilities are computed as
        /// 1. / (1. + np.exp(-self.decision_function(X)));
        /// multiclass is handled by normalizing that over all classes.
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        public Matrix PredictProbaLr(Matrix X)
        {
            var prob = this.DecisionFunction(X);
            prob.MapInplace( v => 1.0 / (Math.Exp(-v) + 1));

            if (prob.ColumnCount == 1)
            {
                var p1 = (Matrix)prob.Clone();
                p1.MapInplace(v => 1 - v);
                return p1.HStack(prob);
            }
            else
            {
                // OvR normalization, like LibLinear's predict_probability
                prob.DivColumnVector(prob.SumColumns(), prob);
                return prob;
            }
        }

        /// <summary>
        /// Predict class labels for samples in X.
        /// </summary>
        /// <param name="x">[n_samples, n_features] Samples.</param>
        /// <returns>[n_samples] Predicted class label per sample.</returns>
        public TLabel[] Predict(Matrix x)
        {
            var scores = this.DecisionFunction(x);
            if (scores.ColumnCount == 1)
                return scores.Column(0).Select(v => v > 0 ? Classes[1] : Classes[0]).ToArray();
            else
                return scores.RowEnumerator().Select(r => Classes[r.Item2.MaximumIndex()]).ToArray();
        }

        public abstract TLabel[] Classes { get; }
        
        protected readonly ClassWeight<TLabel> class_weight;
    }
}
