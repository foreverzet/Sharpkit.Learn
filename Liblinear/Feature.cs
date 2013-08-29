// -----------------------------------------------------------------------
// <copyright file="FeatureNode.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class Feature
    {
        public Feature(int index, double value)
        {
            if (index < 0) throw new ArgumentException("index must be >= 0");
            this.Index = index;
            this.Value = value;
        }

        /// <summary>
        /// Since 1.9
        /// </summary>
        public int Index { get; private set; }

        /// <summary>
        /// Since 1.9
        /// </summary>
        public double Value { get; set; }

        public bool Equals(Feature other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return other.Index == Index && other.Value.Equals(Value);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != typeof (Feature)) return false;
            return Equals((Feature)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (Index*397) ^ Value.GetHashCode();
            }
        }

        public override string ToString()
        {
            return "FeatureNode(idx=" + Index + ", value=" + Value + ")";
        }
    }
}
