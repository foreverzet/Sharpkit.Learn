// -----------------------------------------------------------------------
// <copyright file="StringExtensions.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// <copyright file="StringExtensions.cs" company="Sharpkit.Learn">
//  Copyright Sergey Zyuzin 2014.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn
{
    using System;

    /// <summary>
    /// String extension methods to make porting Python code easier.
    /// </summary>
    internal static class StringExtensions
    {
        public static string Frmt(this string s, params object[] args)
        {
            return string.Format(s, args);
        }
    }
}
