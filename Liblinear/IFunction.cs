// -----------------------------------------------------------------------
// <copyright file="Function.cs" company="">
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
    interface IFunction
    {


        double fun(double[] w);


        void grad(double[] w, double[] g);


        void Hv(double[] s, double[] Hs);


        int get_nr_variable();
    }

}
