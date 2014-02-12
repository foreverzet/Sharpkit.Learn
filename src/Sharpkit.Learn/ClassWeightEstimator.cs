// -----------------------------------------------------------------------
// <copyright file="ClassWeightEstimator.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// Determines the way class weights are calculated.
    /// </summary>
    /// <typeparam name="TLabel">Type of class label.</typeparam>
    public class ClassWeightEstimator<TLabel>
    {
        /// <summary>
        /// Class weights will be given inverse proportional
        /// to the frequency of the class in the data.
        /// </summary>
        public static readonly ClassWeightEstimator<TLabel> Auto = new ClassWeightEstimator<TLabel>(
            "Auto",
            (classes, yInd) => ComputeClassWeightAuto(classes, yInd));
        
        /// <summary>
        /// All class weights will be 1.0.
        /// </summary>
        public static readonly ClassWeightEstimator<TLabel> Uniform = new ClassWeightEstimator<TLabel>(
            "Uniform",
            (classes, yInd) => ComputeClassWeightUniform(classes));

        /// <summary>
        /// Function which computes class weights.
        /// </summary>
        private readonly Func<TLabel[], int[], Vector> func;

        /// <summary>
        /// String representation of the class.
        /// </summary>
        private readonly string textValue = string.Empty;

        /// <summary>
        /// Initializes a new instance of the ClassWeightEstimator class.
        /// </summary>
        /// <param name="textValue">String representation.</param>
        /// <param name="func">Function which computes class weights.</param>
        private ClassWeightEstimator(string textValue, Func<TLabel[], int[], Vector> func)
        {
            this.textValue = textValue;
            this.func = func;
        }

        /// <summary>
        /// Keys are classes, and values
        /// are corresponding class weights.
        /// </summary>
        /// <param name="classWeights">Dictionary which has class weights
        /// corresponding to class labels specified.</param>
        /// <returns>Instance of <see cref="ClassWeightEstimator{TLabel}"/>.</returns>
        public static ClassWeightEstimator<TLabel> Explicit(Dictionary<TLabel, double> classWeights)
        {
            var dictionary = string.Join(",", classWeights.Select(v => string.Format("{0}=>{1}", v.Key, v.Value)));
            return new ClassWeightEstimator<TLabel>(
                string.Format("Explicit({0})", dictionary),
                (classes, y) => ComputeClassWeightExplicit(classWeights, classes));
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>
        /// A string that represents the current object.
        /// </returns>
        public override string ToString()
        {
            return this.textValue;
        }

        /// <summary>
        /// Calculates weights for each sample.
        /// </summary>
        /// <param name="classes">List of all classes.</param>
        /// <param name="yInd">
        /// Target values specified as indixes pointing into <paramref name="classes"/> array.</param>
        /// <returns>
        /// Vector with every element containing weight for every item in <paramref name="yInd"/>.
        /// </returns>
        internal Vector ComputeWeights(TLabel[] classes, int[] yInd)
        {
            return this.func(classes, yInd);
        }

        /// <summary>
        /// Estimate class weights for unbalanced datasets.
        /// </summary>
        /// <param name="classes">Sorted array of the classes occurring in the data.</param>
        /// <param name="yInd">Array of class indices per sample.</param>
        /// <returns>Array with ith element - the weight for i-th class (as determined by sorting).</returns>
        private static Vector ComputeClassWeightAuto(TLabel[] classes, int[] yInd)
        {
            // inversely proportional to the number of samples in the class
            var histogram = new Dictionary<TLabel, int>();
            foreach (var ind in yInd)
            {
                int val;
                histogram.TryGetValue(classes[ind], out val);
                val++;
                histogram[classes[ind]] = val;
            }

            Vector weight = new DenseVector(classes.Count());
            for (int i = 0; i < classes.Count(); i++)
            {
                weight[i] = 1.0 / (histogram.ContainsKey(classes[i]) ? histogram[classes[i]] : 1);
            }

            weight.Multiply(1.0 * classes.Length / weight.Sum(), weight);
            return weight;
        }

        /// <summary>
        /// Computes uniform class weights.
        /// </summary>
        /// <param name="classes">List of all classes.</param>
        /// <returns>Class weights for all classes.</returns>
        private static Vector ComputeClassWeightUniform(TLabel[] classes)
        {
            // uniform class weights
            return DenseVector.Create(classes.Length, i => 1.0);
        }

        /// <summary>
        /// Computes class weights given dictionary which contains weights for certain class labels.
        /// </summary>
        /// <param name="classWeight">Dictionary with class weights.</param>
        /// <param name="classes">List of all classes.</param>
        /// <returns>Vector with weights corresponding to all items in <paramref name="classes"/>.</returns>
        private static Vector ComputeClassWeightExplicit(Dictionary<TLabel, double> classWeight, TLabel[] classes)
        {
            Vector weight = DenseVector.Create(classes.Length, i => 1.0);

            foreach (TLabel c in classWeight.Keys)
            {
                int i = Array.BinarySearch(classes, c);

                if (i < 0)
                {
                    throw new ArgumentException(string.Format("Class label {0} not present.", c));
                }
                else
                {
                    weight[i] = classWeight[c];
                }
            }

            return weight;
        }
    }
}
