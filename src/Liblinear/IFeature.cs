// -----------------------------------------------------------------------
// <copyright file="Feature.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public interface IFeature
    {


        int getIndex();


        double getValue();


        void setValue(double value);
    }

}
