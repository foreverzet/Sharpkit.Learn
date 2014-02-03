// -----------------------------------------------------------------------
// <copyright file="StringExtensions.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    internal static class StringExtensions
    {
        public static string Frmt(this string s, params object[] args)
        {
            return string.Format(s, args);
        }
    }
}
