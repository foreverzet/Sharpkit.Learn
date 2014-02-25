// -----------------------------------------------------------------------
// <copyright file="ILearntSelectorExtensions.cs" company="Sharpkit.Learn">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.FeatureSelection
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Transformer mixin selecting features based on importance weights.
    /// </summary>
    public static class ILearntSelectorExtensions
    {
         /// <summary>
        /// Reduce X to its most important features.
        /// </summary>
        /// <param name="selector"></param>
        /// <param name="x">[n_samples, n_features] The input samples.</param>
        /// <param name="threshold"> The threshold value to use for feature selection. Features whose
        ///        importance is greater or equal are kept while the others are
        ///        discarded.</param>
        /// <returns>[n_samples, n_selected_features]
        ///        The input samples with only the selected features</returns>
        public static Matrix<double> Transform(
            this ILearntSelector selector,
            double[,] x,
            ThresholdChoice threshold)
         {
             return Transform(selector, x.ToDenseMatrix(), threshold);
         }

        /// <summary>
        /// Reduce X to its most important features.
        /// </summary>
        /// <param name="selector"></param>
        /// <param name="x">[n_samples, n_features] The input samples.</param>
        /// <param name="threshold"> The threshold value to use for feature selection. Features whose
        ///        importance is greater or equal are kept while the others are
        ///        discarded.</param>
        /// <returns>[n_samples, n_selected_features]
        ///        The input samples with only the selected features</returns>
        public static Matrix<double> Transform(
            this ILearntSelector selector,
            Matrix<double> x,
            ThresholdChoice threshold)
        {
            if (threshold == null)
            {
                throw new ArgumentNullException("threshold");
            }

            // Retrieve importance vector
            var importances = selector.FeatureImportances();
            if (importances == null)
            {
                throw new InvalidOperationException(
                    "Importance weights not computed. Please set" +
                    " the compute_importances parameter before " +
                    "fit.");
            }
            /*
        elif hasattr(self, "coef_"):
            if self.coef_.ndim == 1:
                importances = np.abs(self.coef_)

            else:
                importances = np.sum(np.abs(self.coef_), axis=0)

        else:
            raise ValueError("Missing `feature_importances_` or `coef_`"
                             " attribute, did you forget to set the "
                             "estimator's parameter to compute it?")
         */
            if (importances.Count != x.ColumnCount)
            {
                throw new InvalidOperationException("X has different number of features than" +
                                                    " during model fitting.");
            }

            /*
        // Retrieve threshold
        if threshold is None:
            if hasattr(self, "penalty") and self.penalty == "l1":
                # the natural default threshold is 0 when l1 penalty was used
                threshold = getattr(self, "threshold", 1e-5)
            else:
                threshold = getattr(self, "threshold", "mean")
        */

            double thresholdvalue = threshold.GetValue(importances);

            var mask =
                importances.GetIndexedEnumerator().Where(v => v.Item2 >= thresholdvalue).Select(v => v.Item1).ToArray();

            if (mask.Any())
            {
                return x.ColumnsAt(mask);
            }
            else
            {
                throw new InvalidOperationException("Invalid threshold: all features are discarded.");
            }
        }
    }
}
