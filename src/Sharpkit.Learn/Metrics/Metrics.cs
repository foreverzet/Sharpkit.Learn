// -----------------------------------------------------------------------
// <copyright file="Metrics.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Metrics
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using Sharpkit.Learn.Utils;

    /// <summary>
    /// Score functions, performance metrics
    /// and pairwise metrics and distance computations.
    /// </summary>
    public static class Metrics
    {
        /// <summary>
        /// RВІ (coefficient of determination) regression score function.
        /// Best possible score is 1.0, lower values are worse.
        /// </summary>
        /// <param name="yTrue">[n_samples, n_outputs]
        /// Ground truth (correct) target values.</param>
        /// <param name="yPred">[n_samples, n_outputs]
        /// Estimated target values.</param>
        /// <returns>The RВІ score.</returns>
        /// <remarks>
        /// <para>
        /// This is not a symmetric function.
        /// Unlike most other scores, RВІ score may be negative (it need not actually
        /// be the square of a quantity R).
        /// </para>
        /// <para>
        /// References
        /// ----------
        /// [1] `Wikipedia entry on the Coefficient of determination
        ///    http://en.wikipedia.org/wiki/Coefficient_of_determination
        /// </para>
        /// </remarks>
        /// <example>
        ///  var yTrue = new[]{ 3, -0.5, 2, 7 };
        ///  var yPred = new[]{ 2.5, 0.0, 2, 8 };
        ///  Metrics.R2Score(yTrue, yPred)
        ///     0.948...
        /// </example>
        public static double R2Score(Matrix<double> yTrue, Matrix<double> yPred)
        {
            if (yTrue.RowCount != yPred.RowCount || yTrue.ColumnCount != yPred.ColumnCount)
            {
                throw new ArgumentException("Dimensions don't match");
            }

            if (yTrue.RowCount <= 1)
            {
                throw new ArgumentException("r2_score can only be computed given more than one sample.");
            }

            double numerator = (yTrue - yPred).Sqr().Sum();
            double denominator = yTrue.SubtractRowVector(yTrue.MeanOfEveryColumn()).Sqr().Sum();

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

            return 1 - (numerator / denominator);
        }

        /// <summary>
        /// <para>
        /// Compute the F1 score, also known as balanced F-score or F-measure
        /// </para>
        /// <para>
        /// The F1 score can be interpreted as a weighted average of the precision and
        /// recall, where an F1 score reaches its best value at 1 and worst score at 0.
        /// The relative contribution of precision and recall to the F1 score are
        /// equal. The formula for the F1 score is::
        /// </para>
        /// <para>
        /// F1 = 2 * (precision * recall) / (precision + recall)
        /// </para>
        /// <para>
        /// In the multi-class and multi-label case, this is the weighted average of
        /// the F1 score of each class.
        /// </para>
        /// <para>
        ///    References
        ///    ----------
        ///    Wikipedia entry for the F1-score
        ///    http://en.wikipedia.org/wiki/F1_score
        /// </para>
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <param name="posLabel">If classification target is binary,
        /// only this class's scores will be returned.</param>
        /// <param name="average">Unless ``posLabel`` is given in binary classification, this
        /// determines the type of averaging performed on the data.</param>
        /// <returns>Weighted average of the F1 scores of each class for the multiclass task.
        /// </returns>
        /// <example>
        ///  In the binary case:
        ///
        ///   var yPred = new[] { 0, 1, 0, 0 };
        ///   var yTrue = new[] { 0, 1, 0, 1 };
        ///   Metrics.F1ScoreAvg(yTrue, yPred);
        ///       0.666...
        ///  
        ///  In multiclass case:
        /// 
        ///  var yTrue = new[] { 0, 1, 2, 0, 1, 2 };
        ///  var yPred = new[] { , 2, 1, 0, 0, 1 };
        ///  Metrics.F1ScoreAvg(yTrue, yPred, average : AverageKind.Micro)
        ///  0.26...
        ///  Metrics.F1ScoreAvg(yTrue, yPred, average : AverageKind.Macro)
        ///  0.33...
        ///  Metrics.F1ScoreAvg(yTrue, yPred, average : AverageKind.Weighted)
        ///  0.26...
        /// </example>
        public static double F1ScoreAvg(
            int[] yTrue,
            int[] yPred,
            int[] labels = null,
            int posLabel = 1,
            AverageKind average = AverageKind.Weighted)
        {
            return FBetaScoreAvg(
                yTrue,
                yPred,
                1,
                labels: labels,
                posLabel: posLabel,
                average: average);
        }

        /// <summary>
        /// <para>
        /// Compute the F1 score, also known as balanced F-score or F-measure for each class.
        /// </para>
        /// <para>
        /// The F1 score can be interpreted as a weighted average of the precision and
        /// recall, where an F1 score reaches its best value at 1 and worst score at 0.
        /// The relative contribution of precision and recall to the F1 score are
        /// equal. The formula for the F1 score is::
        /// </para>
        /// <para>
        /// F1 = 2 * (precision * recall) / (precision + recall)
        /// </para>
        /// <para>
        ///     References
        ///    ----------
        /// .. [1] `Wikipedia entry for the F1-score
        ///    http://en.wikipedia.org/wiki/F1_score
        /// </para>
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <returns>[nUniqueLabels] F1 score for each class.
        /// </returns>
        /// <example>
        ///
        ///  In the multiclass case:
        ///
        ///  var yTrue = new[] { 0, 1, 2, 0, 1, 2 };
        ///  var yPred = new[] { 0, 2, 1, 0, 0, 1 };
        ///  F1Score(yTrue, yPred)
        /// array([ 0.8,  0. ,  0. ])
        /// </example>
        public static double[] F1Score(
            int[] yTrue,
            int[] yPred,
            int[] labels = null)
        {
            return FBetaScore(
                yTrue,
                yPred,
                1,
                labels: labels);
        }

        /// <summary>
        /// Compute the F-beta score
        ///
        /// The F-beta score is the weighted harmonic mean of precision and recall,
        /// reaching its optimal value at 1 and its worst value at 0.
        ///
        /// The `beta` parameter determines the weight of precision in the combined
        /// score. ``beta &lt; 1`` lends more weight to precision, while ``beta > 1``
        /// favors precision (``beta == 0`` considers only precision, ``beta == inf``
        /// only recall).
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="beta">Weight of precision in harmonic mean.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <param name="posLabel">If the classification target is binary,
        /// only this class's scores will be returned.</param>
        /// <param name="average">Unless ``posLabel`` is given in binary classification, this
        /// determines the type of averaging performed on the data.</param>
        /// <returns> F-beta score of the positive class in binary classification or weighted
        /// average of the F-beta score of each class for the multiclass task.</returns>
        /// <remarks>
        ///     References
        ///  ----------
        ///  .. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
        ///   Modern Information Retrieval. Addison Wesley, pp. 327-328.
        ///
        ///   .. [2] `Wikipedia entry for the F1-score
        ///   http://en.wikipedia.org/wiki/F1_score
        /// </remarks>
        /// <example>
        /// In the binary case:
        /// var yPred = new[] { 0, 1, 0, 0 };
        /// var yTrue = new[] { 0, 1, 0, 1 };
        /// Metrics.FBetaScoreAvg(yTrue, yPred, beta: 0.5)
        ///     0.83...
        /// 
        /// Metrics.FBetaScoreAvg(yTrue, yPred, beta: 1)
        ///     0.66...
        /// Metrics.FBetaScoreAvg(yTrue, yPred, beta: 2)
        ///     0.55...
        ///
        /// In the multiclass case:
        ///  
        /// yTrue = new[] { 0, 1, 2, 0, 1, 2 };
        /// yPred = new[] { 0, 2, 1, 0, 0, 1 };
        /// Metrics.FBetaScoreAvg(yTrue, yPred, average: AverageKind.Macro, beta: 0.5);
        ///    0.23...
        /// Metrics.FBetaScoreAvg(yTrue, yPred, average: AverageKind.Micro, beta: 0.5);
        ///    0.33...
        /// Metrics.FBetaScoreAvg(yTrue, y_pred, average: AverageKind.Weighted, beta: 0.5);
        ///    0.23...
        /// </example>
        public static double FBetaScoreAvg(
            int[] yTrue,
            int[] yPred,
            double beta,
            int[] labels = null,
            int posLabel = 1,
            AverageKind average = AverageKind.Weighted)
        {
            var r = PrecisionRecallFScoreSupportAvg(
                yTrue,
                yPred,
                beta: beta,
                labels: labels,
                posLabel: posLabel,
                average: average);

            return r.FBetaScore;
        }

        /// <summary>
        /// Compute the F-beta score for each class.
        ///
        /// The F-beta score is the weighted harmonic mean of precision and recall,
        /// reaching its optimal value at 1 and its worst value at 0.
        ///
        /// The `beta` parameter determines the weight of precision in the combined
        /// score. ``beta &lt; 1`` lends more weight to precision, while ``beta > 1``
        /// favors precision (``beta == 0`` considers only precision, ``beta == inf``
        /// only recall).
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="beta">Weight of precision in harmonic mean.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <returns>F-beta score for each class.</returns>
        /// <remarks>
        ///     References
        ///  ----------
        ///  .. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
        ///   Modern Information Retrieval. Addison Wesley, pp. 327-328.
        ///
        ///   .. [2] `Wikipedia entry for the F1-score
        ///   http://en.wikipedia.org/wiki/F1_score
        /// </remarks>
        /// <example>
        /// In the multiclass case:
        ///  
        /// yTrue = new[]{ 0, 1, 2, 0, 1, 2 };
        /// yPred = new[] { 0, 2, 1, 0, 0, 1 };
        /// Metrics.FBetaScore(yTrue, yPred, beta=0.5)
        ///        {0.71...,  0.        ,  0.        }
        /// </example>
        public static double[] FBetaScore(
            int[] yTrue,
            int[] yPred,
            double beta,
            int[] labels = null)
        {
            var r = PrecisionRecallFScoreSupport(
                yTrue,
                yPred,
                beta: beta,
                labels: labels);

            return r.FBetaScore;
        }

        /// <summary>
        /// Accuracy classification score.
        ///
        /// In multilabel classification, this function computes subset accuracy:
        /// the set of labels predicted for a sample must *exactly* match the
        /// corresponding set of labels in y_true.
        /// </summary>
        /// <param name="y_true">Ground truth (correct) labels.</param>
        /// <param name="y_pred">Predicted labels, as returned by a classifier.</param>
        /// <param name="normalize">If <c>false</c>, return the number of correctly classified samples.
        /// Otherwise, return the fraction of correctly classified samples.</param>
        /// <returns>If <paramref name="normalize"/> == <c>true</c>, return the correctly classified samples
        ///  else it returns the number of correctly classified samples.</returns>
        /// <remarks>
        /// The best performance is 1 with <paramref name="normalize"/> == <c>true</c> and the number
        /// of samples with <paramref name="normalize"/> == <c>false</c>.
        /// </remarks>
        public static double AccuracyScore<TLabel>(TLabel[] y_true, TLabel[] y_pred, bool normalize = true)
            where TLabel : IEquatable<TLabel>
        {
            /*/// <remarks>
/// In binary and multiclass classification, this function is equal
/// to the <see cref="Metrics.JaccardSimilarityScore"/> function.
/// </remarks>
/// <example>
/// </example>
/// <seealso cref="Metrics.HammingLoss"/>
/// <seealso cref="Metrics.ZeroOneLoss"/>
/// <seealso cref="Metrics.JaccardSimilarityScore"/>*/

            /*
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2


    In the multilabel case with binary indicator format:


    >>> accuracy_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.ones((2, 2)))
    0.5


    and with a list of labels format:


    >>> accuracy_score([(1, ), (3, )], [(1, 2), tuple()])
    0.0


    */


            // Compute accuracy for each possible representation
            var score = y_true.Zip(y_pred, Tuple.Create).Select(t => t.Item1.Equals(t.Item2) ? 1 : 0);

            if (normalize)
            {
                return score.Average();
            }
            else
            {
                return score.Sum();
            }
        }

        /// <summary>
        /// Compute the precision
        ///
        /// The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
        /// true positives and ``fp`` the number of false positives. The precision is
        /// intuitively the ability of the classifier not to label as positive a sample
        /// that is negative.
        ///
        /// The best value is 1 and the worst value is 0.
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <param name="posLabel">classification target is binary,
        /// only this class's scores will be returned.</param>
        /// <param name="average">Unless ``posLabel`` is given in binary classification, this
        /// determines the type of averaging performed on the data.</param>
        /// <returns>
        /// Precision of the positive class in binary classification or weighted
        /// average of the precision of each class for the multiclass task.
        /// </returns>
        /// <example>
        /// In the binary case:
        /// var yPred = new[] { 0, 1, 0, 0 };
        /// var yTrue = new[] { 0, 1, 0, 1 };
        /// Metrics.PrecisionScore(yTrue, yPred)
        ///   1.0
        ///
        ///  In the multiclass case:
        ///   var yTrue = new[] { 0, 1, 2, 0, 1, 2 };
        ///   var yPred = new[] { 0, 2, 1, 0, 0, 1 };
        ///   Metrics.PrecisionScoreAvg(yTrue, yPred, average: AverageKind.Macro);
        ///     0.22...
        ///   Metrics.PrecisionScoreAvg(yTrue, yPred, average: AverageKind.Micro);
        ///     0.33...
        ///   Metrics.PrecisionScoreAvg(yTrue, yPred, average: AverageKind.Weighted);
        ///     0.22...
        ///   Metrics.PrecisionScore(yTrue, yPred)
        ///    { 0.66...,  0.        ,  0.        }
        /// </example>
        public static double PrecisionScoreAvg(
            int[] yTrue,
            int[] yPred,
            int[] labels = null,
            int posLabel = 1,
            AverageKind average = AverageKind.Weighted)
        {
            return PrecisionRecallFScoreSupportAvg(
                yTrue,
                yPred,
                labels: labels,
                posLabel: posLabel,
                average: average).Precision;
        }

        /// <summary>
        /// Compute the precision
        ///
        /// The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
        /// true positives and ``fp`` the number of false positives. The precision is
        /// intuitively the ability of the classifier not to label as positive a sample
        /// that is negative.
        ///
        /// The best value is 1 and the worst value is 0.
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <returns>
        /// Precision of the positive class in binary classification or weighted
        /// average of the precision of each class for the multiclass task.
        /// </returns>
        /// <example>
        ///  In the multiclass case:
        ///   var yTrue = new[] { 0, 1, 2, 0, 1, 2 };
        ///   var yPred = new[] { 0, 2, 1, 0, 0, 1 };
        ///   Metrics.PrecisionScore(yTrue, yPred)
        ///    { 0.66...,  0.        ,  0.        }
        /// </example>
        public static double[] PrecisionScore(
            int[] yTrue,
            int[] yPred,
            int[] labels = null)
        {
            return PrecisionRecallFScoreSupport(
                yTrue,
                yPred,
                labels: labels).Precision;
        }

    /// <summary>
    /// Compute the recall
    ///
    /// The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    /// true positives and ``fn`` the number of false negatives. The recall is
    /// intuitively the ability of the classifier to find all the positive samples.
    /// 
    /// The best value is 1 and the worst value is 0.
    /// </summary>
    /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
    /// <param name="yPred">Estimated targets as returned by a classifier.</param>
    /// <param name="labels">Integer array of labels.</param>
    /// <param name="posLabel">classification target is binary,
    /// only this class's scores will be returned.</param>
    /// <param name="average">Unless ``posLabel`` is given in binary classification, this
    /// determines the type of averaging performed on the data.</param>
    /// <returns>Recall of the positive class in binary classification or weighted
    /// average of the recall of each class for the multiclass task.</returns>
    /// <examples>
    ///    In the binary case:
    ///
    ///    var yPred = new[] { 0, 1, 0, 0 };
    ///    var yTrue = new[] { 0, 1, 0, 1 };
    ///    Metrics.RecallScoreAvg(yTrue, yPred);
    ///       0.5
    ///
    /// In the multiclass case:
    ///
    ///    var yTrue = new[] { 0, 1, 2, 0, 1, 2 };
    ///    var yPred = new[] { 0, 2, 1, 0, 0, 1 };
    ///    Metrics.RecallScoreAvg(yTrue, yPred, average: AverageKind.Macro)
    ///       0.33...
    ///    Metrics.RecallScoreAvg(yTrue, yPred, average: AverageKind.Micro)
    ///       0.33...
    ///    Metrics.RecallScoreAvg(yTrue, yPred, average: AverageKind.Weighted)
    ///       0.33...
    ///    Metrics.RecallScore(yTrue, yPred)
    ///      { 1.,  0.,  0. }
    /// </examples>
    public static double RecallScoreAvg(
        int[] yTrue,
        int[] yPred,
        int[] labels = null,
        int posLabel = 1,
        AverageKind average = AverageKind.Weighted)
    {
        return PrecisionRecallFScoreSupportAvg(
            yTrue,
            yPred,
            labels: labels,
            posLabel: posLabel,
            average: average).Recall;
    }

     /// <summary>
     /// Mean squared error regression loss.
     /// </summary>
     /// <param name="yTrue">[n_samples, n_outputs] Ground truth (correct) target values.</param>
     /// <param name="yPred">[n_samples, n_outputs] Estimated target values.</param>
     /// <returns>Loss.</returns>
     public static double MeanSquaredError(Vector<double> yTrue, Vector<double> yPred)
     {
         return MeanSquaredError(yTrue.ToRowMatrix(), yPred.ToRowMatrix());
     }

     /// <summary>
     /// Mean squared error regression loss.
     /// </summary>
     /// <param name="yTrue">[n_samples, n_outputs] Ground truth (correct) target values.</param>
     /// <param name="yPred">[n_samples, n_outputs] Estimated target values.</param>
     /// <returns>Loss.</returns>
     public static double MeanSquaredError(Matrix<double> yTrue, Matrix<double> yPred)
     {
    /*
    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)  # doctest: +ELLIPSIS
    0.708...

    */

         return (yPred - yTrue).Sqr().Mean();
     }

    /// <summary>
    /// Compute the recall
    ///
    /// The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    /// true positives and ``fn`` the number of false negatives. The recall is
    /// intuitively the ability of the classifier to find all the positive samples.
    /// 
    /// The best value is 1 and the worst value is 0.
    /// </summary>
    /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
    /// <param name="yPred">Estimated targets as returned by a classifier.</param>
    /// <param name="labels">Integer array of labels.</param>
    /// <returns>Recall for each class.</returns>
    /// <examples>
    /// In the multiclass case:
    ///
    ///    var yTrue = new[] { 0, 1, 2, 0, 1, 2 };
    ///    var yPred = new[] { 0, 2, 1, 0, 0, 1 };
    ///    Metrics.RecallScore(yTrue, yPred)
    ///      { 1.,  0.,  0. }
    /// </examples>
        public static double[] RecallScore(
        int[] yTrue,
        int[] yPred,
        int[] labels = null)
    {
        return PrecisionRecallFScoreSupport(
            yTrue,
            yPred,
            labels: labels).Recall;
    }

        /// <summary>
        /// Compute average precision, recall, F-measure and support.
        ///
        /// The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
        /// true positives and ``fp`` the number of false positives. The precision is
        /// intuitively the ability of the classifier not to label as positive a sample
        /// that is negative.
        /// 
        /// The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
        /// true positives and ``fn`` the number of false negatives. The recall is
        /// intuitively the ability of the classifier to find all the positive samples.
        ///
        /// The F-beta score can be interpreted as a weighted harmonic mean of
        /// the precision and recall, where an F-beta score reaches its best
        /// value at 1 and worst score at 0.
        ///
        /// The F-beta score weights recall more than precision by a factor of
        /// ``beta``. ``beta == 1.0`` means recall and precsion are equally important.
        ///
        /// The support is the number of occurrences of each class in ``y_true``.
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="beta">Weight of precision in harmonic mean.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <param name="posLabel">If the classification target is binary,
        /// only this class's scores will be returned.</param>
        /// <param name="average">Unless ``posLabel`` is given in binary classification, this
        /// determines the type of averaging performed on the data.</param>
        /// <returns>Instance of <see cref="PrecisionRecallResultAvg"/>.</returns>
        /// <remarks>
        /// .. [1] `Wikipedia entry for the Precision and recall
        ///    http://en.wikipedia.org/wiki/Precision_and_recall_
        ///
        /// .. [2] `Wikipedia entry for the F1-score
        ///    http://en.wikipedia.org/wiki/F1_score
        ///
        /// .. [3] `Discriminative Methods for Multi-labeled Classification Advances
        ///   in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
        ///   Godbole, Sunita Sarawagi
        ///   http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf
        /// </remarks>
        /// <example>
        ///
        /// In the multiclass case:
        ///
        /// yTrue = new[] { 0, 1, 2, 0, 1, 2 };
        /// yPred = new[] { 0, 2, 1, 0, 0, 1 };
        /// var r = Metrics.PrecisionRecallFscoreSupportAvg(yTrue, yPred, average: AverageKind.Macro);
        ///     (Precision = 0.22, Recall = 0.33, FScore = 0.26)
        /// r = Metrics.PrecisionRecallFscoreSupportAvg(yTrue, yPred, average: AverageKind.Micro);
        ///     (Precision = 0.33, Recall = 0.33, FScore = 0.33)
        /// r = Metrics.PrecisionRecallFscoreSupport(yTrue, yPred, average: AverageKind.Weighted);
        ///     (Precision = 0.22, Recall = 0.33, FScore = 0.26)
        /// </example>
        public static PrecisionRecallResultAvg PrecisionRecallFScoreSupportAvg(
            int[] yTrue,
            int[] yPred,
            double beta = 1.0,
            int[] labels = null,
            int? posLabel = 1,
            AverageKind average = AverageKind.Weighted)
        {
            var r = PrecisionRecallFScoreSupportInternal(yTrue, yPred, beta, labels, posLabel, average);
            return new PrecisionRecallResultAvg
                       {
                           FBetaScore = r.FBetaScore[0],
                           Precision = r.Precision[0],
                           Recall = r.Recall[0]
                       };
        }

        /// <summary>
        /// Compute precision, recall, F-measure and support for each class.
        ///
        /// The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
        /// true positives and ``fp`` the number of false positives. The precision is
        /// intuitively the ability of the classifier not to label as positive a sample
        /// that is negative.
        /// 
        /// The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
        /// true positives and ``fn`` the number of false negatives. The recall is
        /// intuitively the ability of the classifier to find all the positive samples.
        ///
        /// The F-beta score can be interpreted as a weighted harmonic mean of
        /// the precision and recall, where an F-beta score reaches its best
        /// value at 1 and worst score at 0.
        ///
        /// The F-beta score weights recall more than precision by a factor of
        /// ``beta``. ``beta == 1.0`` means recall and precsion are equally important.
        ///
        /// The support is the number of occurrences of each class in ``y_true``.
        /// </summary>
        /// <param name="yTrue">List of labels. Ground truth (correct) target values.</param>
        /// <param name="yPred">Estimated targets as returned by a classifier.</param>
        /// <param name="beta">Weight of precision in harmonic mean.</param>
        /// <param name="labels">Integer array of labels.</param>
        /// <returns>Instance of <see cref="PrecisionRecallResult"/>.</returns>
        /// <remarks>
        /// .. [1] `Wikipedia entry for the Precision and recall
        ///    http://en.wikipedia.org/wiki/Precision_and_recall_
        ///
        /// .. [2] `Wikipedia entry for the F1-score
        ///    http://en.wikipedia.org/wiki/F1_score
        ///
        /// .. [3] `Discriminative Methods for Multi-labeled Classification Advances
        ///   in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
        ///   Godbole, Sunita Sarawagi
        ///   http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf
        /// </remarks>
        /// <example>
        ///     In the binary case:
        ///     yPred = new[]{ 0, 1, 0, 0 };
        ///     yTrue = new[]{ 0, 1, 0, 1 };
        ///     r = Metrics.PrecisionRecallFscoreSupport(yTrue, yPred, beta: 0.5);
        ///     r.Precision
        ///        { 0.66...,  1.        }
        ///     r.Recall 
        ///        { 1. ,  0.5 }
        ///     r.FScore
        ///        { 0.71...,  0.83...}
        ///     r.Support s
        ///        {[2, 2]...}
        /// </example>
        public static PrecisionRecallResult PrecisionRecallFScoreSupport(
            int[] yTrue,
            int[] yPred,
            double beta = 1.0,
            int[] labels = null)
        {
            var r = PrecisionRecallFScoreSupportInternal(yTrue, yPred, beta, labels);
            return new PrecisionRecallResult
                       {
                           FBetaScore = r.FBetaScore,
                           Precision = r.Precision,
                           Recall = r.Recall,
                           Support = r.Support
                       };
        }

        private static PrecisionRecallResult PrecisionRecallFScoreSupportInternal(
            int[] yTrue,
            int[] yPred,
            double beta = 1.0,
            int[] labels = null,
            int? posLabel = 1,
            AverageKind? average = null)
        {
            if (beta <= 0)
            {
                throw new ArgumentException("beta should be >0 in the F-beta score");
            }

            var beta2 = beta * beta;

            string yType = CheckClfTargets(yTrue, yPred);

            if (labels == null)
            {
                labels = Multiclass.unique_labels(yTrue, yPred);
            }

            var r = TpTnFpFn(yTrue, yPred, labels);
            var truePos = r.Item1;
            var falsePos = r.Item3;
            var falseNeg = r.Item4;
            var support = truePos.Add(falseNeg);

            // precision and recall
            var precision = truePos.Div(truePos.Add(falsePos));
            var recall = truePos.Div(truePos.Add(falseNeg));

                // fbeta score
            var fscore = new double[precision.Length];
            for (int i = 0; i < fscore.Length; i++)
            {
                fscore[i] = (1 + beta2) * precision[i] * recall[i] / ((beta2 * precision[i]) + recall[i]);
                if (double.IsNaN(fscore[i]))
                {
                    fscore[i] = 0;
                }
            }

            if (average == null)
            {
                return new PrecisionRecallResult
                               {
                                   Precision = precision,
                                   FBetaScore = fscore,
                                   Recall = recall,
                                   Support = support
                               };
            }
            else if (yType == "binary" && posLabel.HasValue)
            {
                if (!labels.Contains(posLabel.Value))
                {
                    if (labels.Length == 1)
                    {
                        // Only negative labels
                        return new PrecisionRecallResult
                                   {
                                       FBetaScore = new double[1],
                                       Precision = new double[1],
                                       Recall = new double[1],
                                       Support = new int[1]
                                   };
                    }

                    throw new ArgumentException(
                        string.Format(
                            "pos_label={0} is not a valid label: {1}",
                             posLabel,
                             string.Join(",", labels)));
                }

                int posLabelIdx = Array.IndexOf(labels, posLabel);
                return new PrecisionRecallResult
                           {
                               Precision = new[] { precision[posLabelIdx] },
                               Recall = new[] { recall[posLabelIdx] },
                               FBetaScore = new[] { fscore[posLabelIdx] },
                               Support = new[] { support[posLabelIdx] }
                           };
            }
            else
            {
                double avgPrecision;
                double avgRecall;
                double avgFscore;

                if (average == AverageKind.Micro)
                {
                    avgPrecision = 1.0 * truePos.Sum() / (truePos.Sum() + falsePos.Sum());
                    avgRecall = 1.0 * truePos.Sum() / (truePos.Sum() + falseNeg.Sum());
                    avgFscore = (1 + beta2) * (avgPrecision * avgRecall) /
                                     ((beta2 * avgPrecision) + avgRecall);

                    if (double.IsNaN(avgPrecision))
                    {
                        avgPrecision = 0.0;
                    }

                    if (double.IsNaN(avgRecall))
                    {
                        avgRecall = 0.0;
                    }

                    if (double.IsNaN(avgFscore))
                    {
                        avgFscore = 0.0;
                    }
                }
                else if (average == AverageKind.Macro)
                {
                    avgPrecision = precision.Average();
                    avgRecall = recall.Average();
                    avgFscore = fscore.Average();
                }
                else if (average == AverageKind.Weighted)
                {
                    if (support.All(v => v == 0))
                    {
                        avgPrecision = 0.0;
                        avgRecall = 0.0;
                        avgFscore = 0.0;
                    }
                    else
                    {
                        avgPrecision = precision.Mul(support).Sum() / support.Sum();
                        avgRecall = recall.Mul(support).Sum() / support.Sum();
                        avgFscore = fscore.Mul(support).Sum() / support.Sum();
                    }
                }
                else
                {
                    throw new ArgumentException("Unsupported argument value", "average");
                }

                return new PrecisionRecallResult
                           {
                               Precision = new[] { avgPrecision },
                               Recall = new[] { avgRecall },
                               FBetaScore = new[] { avgFscore },
                               Support = null
                           };
                }
        }

        private static string CheckClfTargets<T>(T[] yTrue, T[] yPred)
        {
            var typeTrue = Multiclass.type_of_target(yTrue);
            var typePred = Multiclass.type_of_target(yPred);

            if (new[] { "multiclass", "binary" }.Contains(typePred) &&
                new[] { "multiclass", "binary" }.Contains(typeTrue))
            {
                if (new[] { typeTrue, typePred }.Contains("multiclass"))
                {
                    // 'binary' can be removed
                    typeTrue = "multiclass";
                }
            }

            return typeTrue;
        }

        /*
         *     """Compute the number of true/false positives/negative for each class

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or list of labels or label indicator matrix
        Predicted labels, as returned by a classifier.

    labels : array, shape = [n_labels], optional
        Integer array of labels.

    Returns
    -------
    true_pos : array of int, shape = [n_unique_labels]
        Number of true positives

    true_neg : array of int, shape = [n_unique_labels]
        Number of true negative

    false_pos : array of int, shape = [n_unique_labels]
        Number of false positives

    false_pos : array of int, shape = [n_unique_labels]
        Number of false positives

    Examples
    --------
    In the binary case:

    >>> from sklearn.metrics.metrics import _tp_tn_fp_fn
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 0, 1]
    >>> _tp_tn_fp_fn(y_true, y_pred)
    (array([2, 1]), array([1, 2]), array([1, 0]), array([0, 1]))

    In the multiclass case:
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 2, 1, 0, 0, 1])
    >>> _tp_tn_fp_fn(y_true, y_pred)
    (array([2, 0, 0]), array([3, 2, 3]), array([1, 2, 1]), array([0, 2, 2]))

    In the multilabel case with binary indicator format:

    >>> _tp_tn_fp_fn(np.array([[0.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2)))
    (array([0, 0]), array([1, 0]), array([0, 0]), array([1, 2]))

    and with a list of labels format:

    >>> _tp_tn_fp_fn([(1, 2), (3,)], [(1, 2), tuple()])  # doctest: +ELLIPSIS
    (array([1, 1, 0]), array([1, 1, 1]), array([0, 0, 0]), array([0, 0, 1]))

    """

         */
        private static Tuple<int[], int[], int[], int[]> TpTnFpFn(int[] yTrue, int[] yPred, int[] labels = null)
        {
            if (labels == null)
            {
                labels = Multiclass.unique_labels(yTrue, yPred);
            }

            int nLabels = labels.Length;
            var truePos = new int[nLabels];
            var falsePos = new int[nLabels];
            var falseNeg = new int[nLabels];
            var trueNeg = new int[nLabels];

            for (int i = 0; i < labels.Length; i++)
            {
                var labelI = labels[i];
                truePos[i] = yPred
                    .ElementsAt(yTrue.Indices(v => v.Equals(labelI)))
                    .Indices(v => v.Equals(labelI))
                    .Count();

                trueNeg[i] = yPred
                    .ElementsAt(yTrue.Indices(v => !v.Equals(labelI)))
                    .Indices(v => !v.Equals(labelI))
                    .Count();

                falsePos[i] = yPred
                    .ElementsAt(yTrue.Indices(v => !v.Equals(labelI)))
                    .Indices(v => v.Equals(labelI))
                    .Count();

                falseNeg[i] = yPred
                    .ElementsAt(yTrue.Indices(v => v.Equals(labelI)))
                    .Indices(v => !v.Equals(labelI))
                    .Count();
            }

            return Tuple.Create(truePos, trueNeg, falsePos, falseNeg);
        }

        private static int[] Add(this int[] left, int[] right)
        {
            return left.Zip(right, Tuple.Create).Select(t => t.Item1 + t.Item2).ToArray();
        }

        private static double[] Mul(this double[] left, int[] right)
        {
            return left.Zip(right, Tuple.Create).Select(t => t.Item1 * t.Item2).ToArray();
        }

        private static double[] Div(this int[] left, int[] right)
        {
            return left.Zip(right, Tuple.Create)
                .Select(t => t.Item2 == 0 ? 0.0 : (double)t.Item1 / t.Item2).ToArray();
        }
    }
}
