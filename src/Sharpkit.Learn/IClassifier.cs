// -----------------------------------------------------------------------
// <copyright file="IClassifier.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Interface implemented by all classifiers.
    /// </summary>
    /// <typeparam name="TLabel">Type of class label.</typeparam>
    public interface IClassifier<TLabel>
    {
        /// <summary>
        /// Gets or sets class weight estimator.
        /// </summary>
        ClassWeightEstimator<TLabel> ClassWeightEstimator { get; set; }
        
        /// <summary>
        /// Gets ordered list of class labeled discovered int <see cref="Fit"/>.
        /// </summary>
        TLabel[] Classes { get; }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Training vectors,
        /// where nSamples is the number of samples and nFeatures
        /// is the number of features.</param>
        /// <param name="y">[nSamples] Target class labels.</param>
        /// <returns>Reference to itself.</returns>
        IClassifier<TLabel> Fit(Matrix<double> x, TLabel[] y);
        
        /// <summary>
        /// Perform classification on samples in X.
        /// For an one-class model, +1 or -1 is returned.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>[nSamples] Class labels for samples in <paramref name="x"/>.</returns>
        TLabel[] Predict(Matrix<double> x);
        
        /// <summary>
        /// Calculates probability estimates.
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>
        /// [nSamples, nClasses]. The probability of the sample for each class in the model,
        /// where classes are ordered as they are in <see cref="Classes"/>.
        /// </returns>
        Matrix<double> PredictProba(Matrix<double> x);

        /// <summary>
        /// Calculates log of probability estimates.
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Samples.</param>
        /// <returns>[nSamples, nClasses] Log-probability of the sample for each class in the
        /// model, where classes are ordered as they are in <see cref="Classes"/>.
        /// </returns>
        Matrix<double> PredictLogProba(Matrix<double> x);

        /// <summary>
        /// Predict confidence scores for samples.
        /// The confidence score for a sample is the signed distance of that
        /// sample to the hyperplane.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures] Samples.</param>
        /// <returns>[nSamples, nClasses] Confidence scores per (sample, class) combination.
        /// In the binary case, confidence score for the "positive" class.</returns>
        Matrix<double> DecisionFunction(Matrix<double> x);
    }
}
