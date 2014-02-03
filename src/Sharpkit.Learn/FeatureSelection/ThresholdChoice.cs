// -----------------------------------------------------------------------
// <copyright file="ThresholdChoice.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Generic;
    using MathNet.Numerics.Statistics;

    /// <summary>
    /*The threshold value to use for feature selection. Features whose
            importance is greater or equal are kept while the others are
            discarded. If "median" (resp. "mean"), then the threshold value is
            the median (resp. the mean) of the feature importances. A scaling
            factor (e.g., "1.25*mean") may also be used. If None and if
            available, the object attribute ``threshold`` is used. Otherwise,
            "mean" is used by default.
    */
    /// </summary>
    public class ThresholdChoice
    {
        private readonly string name;
        private readonly double value;
        private readonly double scale;

        public static ThresholdChoice Mean(double scale = 1.0)
        {
            return new ThresholdChoice("mean", scale, 1.0);
        }

        public static ThresholdChoice Median(double scale = 1.0)
        {
            return new ThresholdChoice("median", scale, 1.0);
        }

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
