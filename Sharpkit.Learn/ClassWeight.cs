// -----------------------------------------------------------------------
// <copyright file="ClassWeight.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class ClassWeight<TLabel>
    {
        /// <summary>
        /// Class weights will be given inverse proportional
        /// to the frequency of the class in the data.
        /// </summary>
        public static readonly ClassWeight<TLabel> Auto = new ClassWeight<TLabel>(
            (classes, yInd) => ComputeClassWeightAuto(classes, yInd));
        
        public static readonly ClassWeight<TLabel> Uniform = new ClassWeight<TLabel>(
            (classes, yInd) => ComputeClassWeightUniform(classes));
        
        /// <summary>
        /// Keys are classes and values
        /// are corresponding class weights.
        /// </summary>
        /// <param name="classWeights"></param>
        /// <returns></returns>
        public static ClassWeight<TLabel> Explicit(Dictionary<TLabel, double> classWeights)
        {
            return new ClassWeight<TLabel>((classes, y) => ComputeClassWeightExplicit(classWeights, classes));
        }
        
        private readonly Func<TLabel[], int[], Vector> func;
        private ClassWeight(Func<TLabel[], int[], Vector> func)
        {
            this.func = func;
        }

        public Vector ComputeWeights(TLabel[] classes, int[] yInd)
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
            Vector weight;
            // inversely proportional to the number of samples in the class
            var histogram = new Dictionary<TLabel, int>();
            foreach (var ind in yInd)
            {
                int val;
                histogram.TryGetValue(classes[ind], out val);
                val++;
                histogram[classes[ind]] = val;
            }

            weight = new DenseVector(classes.Count());
            for (int i = 0; i < classes.Count(); i++)
            {
                weight[i] = 1.0 / (histogram.ContainsKey(classes[i]) ? histogram[classes[i]] : 1);
            }

            weight.Multiply(1.0 * classes.Length / weight.Sum(), weight);
            return weight;
        }

        private static Vector ComputeClassWeightUniform(TLabel[] classes)
        {
            // uniform class weights
            return DenseVector.Create(classes.Length, i => 1.0);
        }

        private static Vector ComputeClassWeightExplicit(Dictionary<TLabel, double> classWeight, TLabel[] classes)
        {
            Vector  weight = DenseVector.Create(classes.Length, i => 1.0);

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
