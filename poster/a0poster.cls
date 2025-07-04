%$Id: a0poster.cls,v 1.5 1999/04/05 14:49:56 mjf Exp $
%% 
%% This is file `a0poster.cls'
%% 
%% Copyright (C) 07.05.97 Gerlinde Kettl and Matthias Weiser
%%
%% Problems, bugs and comments to 
%% gerlinde.kettl@physik.uni-regensburg.de
%% 
%% changed textwidths and margins to cope with printable area and
%% frame.  Make sure that the offsets are set to -1in!
%% Also changed scaling for a0->a4, since this was wrong.

%% This version changed by Hugh Pumphrey on 4.5.1999. Every instance
%% of the string ``draft'' replaced by the string ``preview''. This gives 
%% an a4 size preview but includes the postscript figures

\ProvidesClass{a0poster}[1997/05/07 v1.21b a0poster class (GK, MW)]
\NeedsTeXFormat{LaTeX2e}[1995/06/01]
\LoadClass{article}

\newif\ifportrait
\newif\ifanullb
\newif\ifanull
\newif\ifaeins
\newif\ifazwei
\newif\ifadrei
\newif\ifpreview

\newcount\xkoord
\newcount\ykoord
\newcount\xscale
\newcount\yscale

\DeclareOption{a0b}{
\anullbtrue
\xkoord=2592 % big points (1 bp=1/72 inch)
\ykoord=3666 % big points (1 bp=1/72 inch)
\xscale=23
\yscale=23
}
\DeclareOption{a0}{
\anulltrue\anullbfalse
\xkoord=2380 % big points (1 bp=1/72 inch)
\ykoord=3368 % big points (1 bp=1/72 inch)
\xscale=25
\yscale=25
}
\DeclareOption{a1}{
\aeinstrue\anullbfalse
\xkoord=1684 % big points (1 bp=1/72 inch)
\ykoord=2380 % big points (1 bp=1/72 inch)
\xscale=3
\yscale=3
}
\DeclareOption{a2}{
\azweitrue\anullbfalse
\xkoord=1190 % big points (1 bp=1/72 inch)
\ykoord=1684 % big points (1 bp=1/72 inch)
\xscale=4
\yscale=4

}
\DeclareOption{a3}{
\adreitrue\anullbfalse
\xkoord=842  % big points (1 bp=1/72 inch)
\ykoord=1190 % big points (1 bp=1/72 inch)
\xscale=6
\yscale=6
}
\DeclareOption{landscape}{
\portraitfalse
}
\DeclareOption{portrait}{
\portraittrue
}
\DeclareOption{preview}{
\previewtrue
}
\DeclareOption{final}{
\previewfalse
}
\DeclareOption*{\PackageWarning{a0poster}{Unknown Option \CurrentOption}}
\ExecuteOptions{landscape,a0b,final}
\ProcessOptions\relax


% MY CODE------------------

\def\thebibliography#1{\section*{References}
  \global\def\@listi{\leftmargin\leftmargini
               \labelwidth\leftmargini \advance\labelwidth-\labelsep
               \topsep 1pt plus 2pt minus 1pt
               \parsep 0.25ex plus 1pt \itemsep 0.25ex plus 1pt}
  \list {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
    \advance\leftmargin\labelsep\usecounter{enumi}}
    \def\newblock{\hskip .11em plus .33em minus -.07em}
    \sloppy
    \sfcode`\.=1000\relax}

\def\@up#1{\raise.2ex\hbox{#1}}

% don't box citations, separate with ; and a space
% Replaced for multiple citations (pj)
% don't box citations and also add space, semicolon between multiple citations

\def\@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
  \def\@citea{}\@cite{\@for\@citeb:=#2\do
     {\@citea\def\@citea{; }\@ifundefined
       {b@\@citeb}{{\bf ?}\@warning
        {Citation `\@citeb' on page \thepage \space undefined}}%
 {\csname b@\@citeb\endcsname}}}{#1}}

% Allow short (name-less) citations, when used in
% conjunction with a bibliography style that creates labels like
%       \citename{<names>, }<year>
%
\let\@internalcite\cite
\def\cite{\def\citename##1{##1, }\@internalcite}
\def\shortcite{\def\citename##1{}\@internalcite}
\def\newcite{\def\citename##1{{\frenchspacing##1} (}\@internalciteb}

% Macros for \newcite, which leaves name in running text, and is
% otherwise like \shortcite.
\def\@citexb[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
  \def\@citea{}\@newcite{\@for\@citeb:=#2\do
    {\@citea\def\@citea{;\penalty\@m\ }\@ifundefined
       {b@\@citeb}{{\bf ?}\@warning
       {Citation `\@citeb' on page \thepage \space undefined}}%
{\csname b@\@citeb\endcsname}}}{#1}}
\def\@internalciteb{\@ifnextchar [{\@tempswatrue\@citexb}{\@tempswafalse\@citexb[]}}

\def\@newcite#1#2{{#1\if@tempswa, #2\fi)}}

\def\@biblabel#1{\def\citename##1{##1}[#1]\hfill}

%%% More changes made by SMS (originals in latex.tex)
% Use parentheses instead of square brackets in the text.
\def\@cite#1#2{({#1\if@tempswa , #2\fi})}

% Don't put a label in the bibliography at all.  Just use the unlabeled format
% instead.
\def\thebibliography#1{\vskip\parskip%
\vskip\baselineskip%
\def\baselinestretch{1}%
\ifx\@currsize\normalsize\@normalsize\else\@currsize\fi%
\vskip-\parskip%
\vskip-\baselineskip%
\section*{References\@mkboth
 {References}{References}}\list
 {}{\setlength{\labelwidth}{0pt}\setlength{\leftmargin}{\parindent}
 \setlength{\itemindent}{-\parindent}}
 \def\newblock{\hskip .11em plus .33em minus -.07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

% Allow for a bibliography of sources of attested examples
\def\thesourcebibliography#1{\vskip\parskip%
\vskip\baselineskip%
\def\baselinestretch{1}%
\ifx\@currsize\normalsize\@normalsize\else\@currsize\fi%
\vskip-\parskip%
\vskip-\baselineskip%
\section*{Sources of Attested Examples\@mkboth
 {Sources of Attested Examples}{Sources of Attested Examples}}\list
 {}{\setlength{\labelwidth}{0pt}\setlength{\leftmargin}{\parindent}
 \setlength{\itemindent}{-\parindent}}
 \def\newblock{\hskip .11em plus .33em minus -.07em}
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}
\let\endthesourcebibliography=\endlist

\def\@lbibitem[#1]#2{\item[]\if@filesw
      { \def\protect##1{\string ##1\space}\immediate
        \write\@auxout{\string\bibcite{#2}{#1}}\fi\ignorespaces}}

\def\@bibitem#1{\item\if@filesw \immediate\write\@auxout
       {\string\bibcite{#1}{\the\c@enumi}}\fi\ignorespaces}
%------------------

\ifanullb
   \setlength{\paperwidth}{129.3cm} %% 36 * sqrt(2) in
   \setlength{\paperheight}{91.4cm} %% 36 in
   \setlength{\textwidth}{119.3cm} %% paperwidth - (5cm + 5cm)
   \setlength{\textheight}{81.4cm} %% paperheight - (5cm + 5cm)
\else\ifanull
        \setlength{\paperwidth}{118.82cm}
        \setlength{\paperheight}{83.96cm}
        \setlength{\textwidth}{108.82cm} %% paperwidth - (5cm + 5cm)
        \setlength{\textheight}{73.96cm} %% paperheight - (5cm + 5cm)
     \else\ifaeins
             \setlength{\paperwidth}{83.96cm}
             \setlength{\paperheight}{59.4cm}
             \setlength{\textwidth}{79.96cm}
             \setlength{\textheight}{55.4cm}
          \else\ifazwei
                  \setlength{\paperwidth}{59.4cm}
                  \setlength{\paperheight}{41.98cm}
                  \setlength{\textwidth}{55.4cm}
                  \setlength{\textheight}{37.98cm}
               \else\ifadrei
                       \setlength{\paperwidth}{41.98cm}
                       \setlength{\paperheight}{29.7cm}
                       \setlength{\textwidth}{37.98cm}
                       \setlength{\textheight}{25.7cm}
                    \else\relax
                    \fi
               \fi
          \fi
     \fi
\fi


\ifportrait
   \newdimen\tausch
   \setlength{\tausch}{\paperwidth}
   \setlength{\paperwidth}{\paperheight}
   \setlength{\paperheight}{\tausch}
   \setlength{\tausch}{\textwidth}
   \setlength{\textwidth}{\textheight}
   \setlength{\textheight}{\tausch}
\else\relax
\fi

%% Setting proper dimensions for the HP2500CP printer (height = 36 in)
%%   Landscape: unprintable areas
%%   	L: 27.6mm
%%   	R: 27.1mm
%%   	T: 18.4mm
%%   	B: 18.1mm


\voffset -1in
\hoffset -1in

\setlength{\headheight}{0 cm}
\setlength{\headsep}{0 cm}
\setlength{\topmargin}{4 cm} %% 3 cm for unprintable at top
			     %% (landscape) + 2 cm from border
\setlength{\oddsidemargin}{4 cm} %% 3 cm for unprintable at left
			    	 %% (landscape) + 2 cm from border
\setlength{\topskip}{0 cm}

\input{a0size.sty}

\renewcommand{\tiny}{\fontsize{12}{14}\selectfont}
\renewcommand{\scriptsize}{\fontsize{14.4}{18}\selectfont}   
\renewcommand{\footnotesize}{\fontsize{17.28}{22}\selectfont}
\renewcommand{\small}{\fontsize{20.74}{25}\selectfont}
\renewcommand{\normalsize}{\fontsize{24.68}{30}\selectfont}
%\renewcommand{\normalsize}{\fontsize{24.88}{30}\selectfont}
%\renewcommand{\normalsize}{\fontsize{25.68}{30.7}\selectfont}
\renewcommand{\large}{\fontsize{29.86}{37}\selectfont}
\renewcommand{\Large}{\fontsize{35.83}{45}\selectfont}
\renewcommand{\LARGE}{\fontsize{43}{54}\selectfont}
\renewcommand{\huge}{\fontsize{51.6}{64}\selectfont}
\renewcommand{\Huge}{\fontsize{61.92}{77}\selectfont}
\newcommand{\veryHuge}{\fontsize{74.3}{93}\selectfont}
\newcommand{\VeryHuge}{\fontsize{89.16}{112}\selectfont}
\newcommand{\VERYHuge}{\fontsize{107}{134}\selectfont}

\setlength\smallskipamount{6pt plus 2pt minus 2pt}
\setlength\medskipamount{12pt plus 4pt minus 4pt}
\setlength\bigskipamount{24pt plus 8pt minus 8pt}

\setlength\abovecaptionskip{25pt}
\setlength\belowcaptionskip{0pt}
\setlength\abovedisplayskip{25pt plus 6pt minus 15 pt}
\setlength\abovedisplayshortskip{0pt plus 6pt}
\setlength\belowdisplayshortskip{13pt plus 7pt minus 6pt}
\setlength\belowdisplayskip\abovedisplayskip

\normalsize

\AtBeginDocument{
\ifanullb
   \ifportrait\special{papersize=91.4cm,129.3cm}\else\special{papersize=129.3cm,91.4cm}\fi
\else\ifanull
        \ifportrait\special{papersize=83.96cm,118.82cm}\else\special{papersize=118.82cm,83.96cm}\fi
     \else\ifaeins
             \ifportrait\special{papersize=59.4cm,83.96cm}\else\special{papersize=83.96cm,59.4cm}\fi
          \else\ifazwei
                  \ifportrait\special{papersize=41.98cm,59.4cm}\else\special{papersize=59.4cm,41.98cm}\fi
               \else\ifadrei
                      \ifpreview
                       \ifportrait\special{papersize=29.7cm,41.35cm}\else\special{papersize=41.98cm,29.0cm}\fi
                      \else
                       \ifportrait\special{papersize=29.7cm,41.98cm}\else\special{papersize=41.98cm,29.7cm}\fi
                      \fi
                    \else\relax
                    \fi
               \fi
          \fi
     \fi
\fi
\pagestyle{empty}}

\endinput
%% 
%% End of file `a0poster.cls'.
