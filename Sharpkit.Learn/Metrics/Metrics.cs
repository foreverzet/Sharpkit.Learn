// -----------------------------------------------------------------------
// <copyright file="Metrics.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Metrics
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// Score functions, performance metrics
    /// and pairwise metrics and distance computations.
    /// </summary>
    public static class Metrics
    {
        /// <summary>
        /// RВІ (coefficient of determination) regression score function.
        ///
        /// Best possible score is 1.0, lower values are worse.
        /// </summary>
        /// <param name="yTrue">[n_samples, n_outputs]
        /// Ground truth (correct) target values.</param>
        /// <param name="yPred">[n_samples, n_outputs]
        /// Estimated target values.</param>
        /// <returns>The RВІ score.</returns>
        /// <remarks>
        /// This is not a symmetric function.
        ///
        /// Unlike most other scores, RВІ score may be negative (it need not actually
        /// be the square of a quantity R).
        ///
        /// References
        /// ----------
        /// [1] `Wikipedia entry on the Coefficient of determination
        ///    <http://en.wikipedia.org/wiki/Coefficient_of_determination>
        /// </remarks>
        /// <example>
        ///  y_true = [3, -0.5, 2, 7]
        ///  y_pred = [2.5, 0.0, 2, 8]
        ///  R2Score(y_true, y_pred)  # doctest: +ELLIPSIS
        ///   0.948...
        ///  y_true = [[0.5, 1], [-1, 1], [7, -6]]
        ///  y_pred = [[0, 2], [-1, 2], [8, -5]]
        ///  R2Score(y_true, y_pred)  # doctest: +ELLIPSIS
        ///     0.938...
        /// </example>
        public static double R2Score(Matrix yTrue, Matrix yPred)
        {
            if (yTrue.RowCount != yPred.RowCount || yTrue.ColumnCount != yPred.ColumnCount)
            {
                throw new ArgumentException("Dimensions don't match");
            }

            if (yTrue.RowCount <= 1)
            {
                throw new ArgumentException("r2_score can only be computed given more than one sample.");
            }

            double numerator = ((Matrix)(yTrue - yPred)).Sqr().Sum();
            double denominator = yTrue.SubtractRowVector(yTrue.MeanRows()).Sqr().Sum();

            if (denominator == 0.0)
            {
                if (numerator == 0.0)
                {
                    return 1.0;
                }

                // arbitrary set to zero to avoid -inf scores, having a constant
                // y_true is not interesting for scoring a regression anyway
                return 0.0;
            }

            return 1 - numerator/denominator;
        }
    }
}
