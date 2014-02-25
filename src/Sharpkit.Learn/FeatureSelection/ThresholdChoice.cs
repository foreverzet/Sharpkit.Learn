// -----------------------------------------------------------------------
// <copyright file="ThresholdChoice.cs" company="Sharpkit.Learn">
//  Copyright (c) 2013 Sergey Zyuzin
//  License: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.FeatureSelection
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.Statistics;

    /// <summary>
    /// The threshold value to use for feature selection. Features whose
    /// importance is greater or equal are kept while the others are
    /// discarded.
    /// </summary>
    public class ThresholdChoice
    {
        private readonly string name;
        private readonly double value;
        private readonly double scale;

         /// <summary>
        /// The threshold value is the mean of the feature importances. A scaling
        /// factor (e.g., "1.25*mean") may also be used.
        /// </summary>
        /// <param name="scale">Scaling factor.</param>
        /// <returns>ThresholdChoice instace.</returns>
        public static ThresholdChoice Mean(double scale = 1.0)
        {
            return new ThresholdChoice("mean", scale, 1.0);
        }

        /// <summary>
        /// The threshold value is the median of the feature importances. A scaling
        /// factor (e.g., "1.25*median") may also be used.
        /// </summary>
        /// <param name="scale">Scaling factor.</param>
        /// <returns>ThresholdChoice instace.</returns>
        public static ThresholdChoice Median(double scale = 1.0)
        {
            return new ThresholdChoice("median", scale, 1.0);
        }

        /// <summary>
        /// Threshold value is specified explicitly.
        /// </summary>
        /// <param name="value">Thershold value.</param>
        /// <returns>ThresholdChoice instace.</returns>
        public static ThresholdChoice Value(double value)
        {
            return new ThresholdChoice("value", 1.0, value);
        }

        private ThresholdChoice(string name, double scale, double value)
        {
            this.name = name;
            this.scale = scale;
            this.value = value;
        }

        /// <summary>
        /// Calculates threshold value.
        /// </summary>
        /// <param name="importances">Feature importances.</param>
        /// <returns>Threshold value.</returns>
        public double GetValue(Vector<double> importances)
        {
            switch(name)
            {
                case "mean":
                    return importances.Mean()*scale;
                case "median":
                    return importances.Median() * scale;
                case "value":
                    return value;
                default:
                    throw new ArgumentException("Invalid value");
            }
        }

    }
}
