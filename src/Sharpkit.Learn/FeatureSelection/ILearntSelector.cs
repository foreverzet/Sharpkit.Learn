// -----------------------------------------------------------------------
// <copyright file="ILearntSelector.cs" company="Sharpkit.Learn">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.FeatureSelection
{
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public interface ILearntSelector
    {
        Vector<double> FeatureImportances();
    }
}
