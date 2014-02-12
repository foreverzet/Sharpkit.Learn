// -----------------------------------------------------------------------
// <copyright file="IClassifierExtensions.cs" company="Sharpkit.Learn">
//  Copyright Sergey Zyuzin 2014.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// <see cref="IClassifier{TLabel}"/> extension methods.
    /// </summary>
    public static class IClassifierExtensions
    {
        /// <summary>
        /// Predict class labels for sample x.
        /// </summary>
        /// <typeparam name="TLabel">Type of label.</typeparam>
        /// <param name="classifier">A <see cref="IClassifier{TLabel}"/> instance.</param>
        /// <param name="x">Sample features. Array [n_features]</param>
        /// <returns>Predicted class label.</returns>
        public static TLabel Predict<TLabel>(this IClassifier<TLabel> classifier, double[] x)
        {
            return classifier.Predict(x.ToDenseVector().ToRowMatrix())[0];
        }

        /// <summary>
        /// Predict class labels for samples in x.
        /// </summary>
        /// <typeparam name="TLabel">Type of label.</typeparam>
        /// <param name="classifier">A <see cref="IClassifier{TLabel}"/> instance.</param>
        /// <param name="x">Samples. Array [n_samples, n_features]</param>
        /// <returns>Predicted class label per sample. Array with length [n_samples]</returns>
        public static TLabel[] Predict<TLabel>(this IClassifier<TLabel> classifier, double[,] x)
        {
            return classifier.Predict(x.ToDenseMatrix());
        }

        /// <summary>
        /// Calculates probability estimates.
        /// The returned estimates for all classes are ordered by the label of classes.
        /// </summary>
        /// <typeparam name="TLabel">Type of label.</typeparam>
        /// <param name="classifier">A <see cref="IClassifier{TLabel}"/> instance.</param>
        /// <param name="x">Feature matrix. Array with dimensions [n_samples, n_features]</param>
        /// <returns>Matrix with dimensions [n_samples, n_classes].
        /// Contains the probability of the sample for each class in the model,
        /// where classes are ordered as they are in <see cref="IClassifier{TLabel}.Classes"/>.
        /// </returns>
        public static Matrix<double> PredictProba<TLabel>(this IClassifier<TLabel> classifier, double[,] x)
        {
            return classifier.PredictProba(x.ToDenseMatrix());
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <typeparam name="TLabel">Type of label.</typeparam>
        /// <param name="classifier">A <see cref="IClassifier{TLabel}"/> instance.</param>
        /// <param name="x">
        /// Array with dimensions [nSamples, nFeatures].
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">Array with dimensions [nSamples]. Class labels.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Array with dimensions [nSamples].</param>
        public static void Fit<TLabel>(
            this IClassifier<TLabel> classifier,
            double[,] x,
            TLabel[] y,
            double[] sampleWeight = null)
        {
            classifier.Fit(x.ToDenseMatrix(), y, sampleWeight == null ? null : DenseVector.OfEnumerable(sampleWeight));
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <typeparam name="TLabel">Type of label.</typeparam>
        /// <param name="classifier">A <see cref="IClassifier{TLabel}"/> instance.</param>
        /// <param name="x">
        /// Matrix with dimensions [nSamples, nFeatures].
        /// Training vectors, where nSamples is the number of samples
        /// and nFeatures is the number of features.</param>
        /// <param name="y">Array with dimensions [nSamples]. Class labels.</param>
        /// <param name="sampleWeight">Individual weights for each sample. Array with dimensions [nSamples].</param>
        public static void Fit<TLabel>(
            this IClassifier<TLabel> classifier,
            Matrix<double> x,
            TLabel[] y,
            double[] sampleWeight = null)
        {
            classifier.Fit(x, y, sampleWeight == null ? null : DenseVector.OfEnumerable(sampleWeight));
        }

        /// <summary>
        /// Returns the mean accuracy on the given test data and labels.
        /// </summary>
        /// <typeparam name="TLabel">Type of label.</typeparam>
        /// <param name="classifier">A <see cref="IClassifier{TLabel}"/> instance.</param>
        /// <param name="x">(n_samples, n_features) Test samples.</param>
        /// <param name="y">(n_samples,) True labels for X.</param>
        /// <returns>Mean accuracy of self.predict(X) wrt. y.</returns>
        public static double Score<TLabel>(this IClassifier<TLabel> classifier, Matrix<double> x, TLabel[] y)
            where TLabel : IEquatable<TLabel>
        {
            return Metrics.Metrics.AccuracyScore(y, classifier.Predict(x));
        }

        /// <summary>
        /// Calculates log of probability estimates.
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <typeparam name="TLabel">Type of label.</typeparam>
        /// <param name="classifier">A <see cref="IClassifier{TLabel}"/> instance.</param>
        /// <param name="x">Samples. Matrix with dimensions [nSamples, nFeatures]</param>
        /// <returns>[nSamples, nClasses] Log-probability of the sample for each class in the
        /// model, where classes are ordered as they are in <see cref="IClassifier{TLabel}.Classes"/>.
        /// </returns>
        public static Matrix<double> PredictLogProba<TLabel>(this IClassifier<TLabel> classifier, Matrix<double> x)
        {
            return classifier.PredictProba(x).Log();
        }
    }
}
