\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{HSEUniversity}[2019-06-15 Template for dessirtations and other artefacts]
%\LoadClass[a4paper]{extreport}
\LoadClass[fontsize=14bp]{scrreprt}
\RequirePackage{ifpdf}
\RequirePackage{kvoptions}
\RequirePackage[a4paper,left=40mm,right=10mm,top=20mm,bottom=20mm,footskip=10mm]{geometry}

% \RequirePackage{cmap}						% Улучшенный поиск русских слов в полученном pdf-файле
\RequirePackage[T2A]{fontenc}				% Поддержка русских букв
% \RequirePackage[utf8]{inputenc}				% Кодировка utf8
% \RequirePackage{csquotes}
\RequirePackage[russian]{babel}	% Языки: русский, английский

\usepackage{fontspec}
\usepackage{polyglossia}
% \setmainfont{Times New Roman}
% \newfontfamily\cyrillicfont{Times New Roman}
\setmainfont{Times New Roman}
\newfontfamily\cyrillicfont{Times New Roman}[Script=Cyrillic]
\newfontfamily{\cyrillicfonttt}{Times New Roman}
\setmainlanguage{russian}

% My code
\usepackage{longtable}
\usepackage{array}
\usepackage{float} % Позволяет расположить таблицу именно там, где она назодится относительно текста
\usepackage{enumitem}
\setlist[itemize]{left=13mm, topsep=0pt, partopsep=0pt, itemsep=0pt, parsep=0pt}
\setlist[enumerate]{left=13mm, topsep=0pt, partopsep=0pt, itemsep=0pt, parsep=0pt}
% ----

\RequirePackage[font=small,skip=4pt]{caption}
\RequirePackage{graphicx}

%\usepackage{pscyr}						% Красивые русские шрифты
% \renewcommand{\rmdefault}{ftm} % Включаем Times New Roman


%%% Оформление абзацев %%%
\RequirePackage{indentfirst} % Красная строка

%%% Размер шрифатов у заголовков %%%
\RequirePackage{titlesec}

%%% Table of Contents %%%
\RequirePackage{tocloft}
\RequirePackage{titletoc}

\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}


\RequirePackage[linktocpage=true,plainpages=false,pdfpagelabels=false]{hyperref}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Библиография %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[
	style=gost-numeric-min,
%	citestyle=gost-authoryear-min,
%    bibstyle=gost-numeric-min,
	language=russian,
	babel=other,
	backend=biber,
%	defernumbers=true,% Нумерация начинается как только встречается группа.
	sorting=none,
	url=false,
	doi=false,
	isbn=false,
	movenames=false,
	maxnames=10
]{biblatex}
\renewcommand*{\multicitedelim}{\addcomma\space}

%\usepackage{cite} % Красивые ссылки на литературу
% Настройки языка в списке литературы.
\DeclareSourcemap{
	\maps[datatype=bibtex,overwrite=false]{
	 \map{
			\step[fieldset=language,fieldvalue=english]
	 }
	\map{
		\step[fieldsource=language,match=\regexp{English},replace=\regexp{english}]
	}
	\map{
		\step[fieldsource=language]
		\step[fieldset=hyphenation,origfieldval]
	}
%   \map{
			%\step[fieldset=hyphenation,fieldvalue={english}]
		%}
	}
}
%%%%%%%%%%%%%%%%%%%%%%
% ошибка в russian.lbx
\NewBibliographyString{langjapanese}
\NewBibliographyString{fromjapanese}

\newcommand{\putbibliography}{
	\chapter**{Библиографический список}
	\addcontentsline{toc}{chapter}{Библиографический список}

	\printbibliography[env=gostbibliography,heading=none] %title={Библиографический список},
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Установка основных характеристик текста

\sloppy				% Избавляемся от переполнений
\clubpenalty=10000		% Запрещаем разрыв страницы после первой строки абзаца
\widowpenalty=10000		% Запрещаем разрыв страницы после последней строки абзаца

\linespread{1.5} % Полуторный интервал
\setlength{\parindent}{12.5mm} % Отступ красной строки

\captionsetup[table]{position=t,singlelinecheck=false, justification=RaggedLeft,labelsep=period}
\captionsetup[figure]{labelfont={bf},textfont={bf},position=b,singlelinecheck=false,labelsep=period,justification=centering}

%%% Главы и разделы %%%
\titleformat{\chapter}%[display]
  {\Large\bfseries\centering}
  {\chaptertitlename\ \thechapter.}{4pt}{}
\titlespacing*{\chapter}{0pt}{0pt}{12pt}
\titleformat{\section}
  {\large\centering}
  {\thesection.}{4pt}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\titleformat{\subsection}
  {\centering}
  {\thesubsection.}{3pt}{}
\titlespacing*{\subsection}{0pt}{8pt}{4pt}
\titleformat{\subsubsection}
  {\centering}
  {\thesubsubsection.}{3pt}{}
\titlespacing*{\subsubsection}{0pt}{8pt}{4pt}

%%% Оглавление %%%
\renewcommand{\cftchapdotsep}{\cftdotsep}
\let\@originalchapter\chapter
\def\chapter{
	 \@ifstar\chapterstar\@originalchapter
 }
\def\chapterstar{
	 \@ifstar\chapter@nonum@notoc\chapter@nonum
 }
\newcommand\chapter@nonum@notoc[2][]{
   \@originalchapter*{#2}
}
\newcommand\chapter@nonum[2][]{
   \@originalchapter*{#2}
   \addcontentsline{toc}{chapter}{#2}
}
\titlecontents{chapter}% <section-type>
  [0pt]% <left>
  {}% <above-code>
  {\bfseries\chaptername\ \thecontentslabel.\enskip}% <numbered-entry-format>
  {}% <numberless-entry-format>
  {\bfseries\dotfill\contentspage}% <filler-page-format>
%%%%%%%%%%%%%%%%%%%%%%%%
% Обработка опций пакета

\SetupKeyvalOptions{
	family=HSE,
	prefix=HSE@
}

\def\@facultyString{UNKNOWN FACULTY}
\def\@typeString{UNKNOWN WORK TYPE}
\def\@specName{UNKOWN SPECIALIZATION}
\def\@specCode{00.00.00.UNKOWN SPECIALIZATION}
\newif\ifHasReviewer
\HasReviewerfalse
\newif\ifHasConsultant
\HasConsultantfalse

% Available options for the specialiazation
% LI -- Фундаментальная и прикладная лингвистика
\DeclareVoidOption{LI}{
	\def\@facultyString{Факультет гуманитарных наук}
	\def\@specName{Фундаментальная и прикладная лингвистика}
	\def\@specCode{45.03.03. Фундаментальная и прикладная лингвистика}
}
% Available options for the type
\DeclareVoidOption{KR}{
	\def\@typeString{Курсовая работа}
}
\DeclareVoidOption{VKR}{
	\def\@typeString{Выпускная квалификационная работа}
}

\DeclareDefaultOption{%
	\PackageWarningNoLine{\@currname}{%
		Unknown option `\CurrentOption'\MessageBreak
		is passed to class `report'%
	}%
	% Pass the option to class report
	\expandafter\PassOptionToClass
	\expandafter{\CurrentOption}{report}%
}
\ProcessKeyvalOptions*

\def\@city{Москва}
\def\@year{1900}
\def\@supervisorTitle{}
\def\@supervisorName{I.I.~Ivanov}
\def\@reviewerTitle{}
\def\@reviewerName{I.I.~Ivanov}
\def\@abstract{TODO}

\newcommand{\Year}[1]{\def\@year{#1}}
\newcommand{\City}[1]{\def\@city{#1}}
\newcommand{\supervisor}[2]{
	\def\@supervisorTitle{#1}
	\def\@supervisorName{#2}
}
\newcommand{\consultant}[2]{
	\def\@consultantTitle{#1}
	\def\@consultantName{#2}
}
\newcommand{\reviewer}[2]{
	\def\@reviewerTitle{#1}
	\def\@reviewerName{#2}
}
\newcommand{\Abstract}[1]{
	\def\@abstract{#1}
}

\renewcommand{\maketitle}{
{
	\thispagestyle{empty}

	\centering

	{
		ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ АВТОНОМНОЕ \\
		ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ \\ 
		«НАЦИОНАЛЬНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ УНИВЕРСИТЕТ \\ 
		«ВЫСШАЯ ШКОЛА ЭКОНОМИКИ»
	}

	\vfill
	\vfill

	\textbf{\@facultyString}

	\vfill

	\textbf{Программа подготовки бакалавров по направлению
  	по направлению \@specCode}

	\vfill
	\vfill

	\@author

	\textbf{\MakeUppercase{\@typeString}}

	\vfill

	\@title

	\vfill
	\vfill
	\vfill

	\ifHasReviewer
		\begin{minipage}[t]{0.4\textwidth}
			Рецензент \\
			\@reviewerTitle

			\vspace{2mm}

			\@reviewerName
		\end{minipage}
	\fi
	\hfill
	\begin{minipage}[t]{0.4\textwidth}
		Руководитель \\
		\@supervisorTitle

		\@supervisorName
	\end{minipage}

	\vspace{10mm}

	\ifHasConsultant
		\hfill
		\begin{minipage}[t]{0.4\textwidth}
			Научный консультант \\
			\@consultantTitle

			\@consultantName
		\end{minipage}
	\fi

	\vfill
	\vfill
	\vfill

	\@city{}, \@year{}

	\newpage
}

%\chapter**{Аннотация}

%\@abstract

\chapter**{Оглавление}

\@starttoc{toc}

\newpage

}

\newsavebox{\TABLEbox}
\newenvironment{TABLE}[2][t]
{
  \def\@tableCaption{#2}
  \begin{table}[#1]
  \begin{lrbox}{\TABLEbox}
  \small
}{
  \end{lrbox}

  \centering
  \captionbox[l]{\@tableCaption}{
    \usebox{\TABLEbox}
  }
  \end{table}
}


\newsavebox{\FIGUREbox}
\newenvironment{FIGURE}[2][t]
{
  \def\@figureCaption{#2}
  \begin{figure}[#1]
  \begin{lrbox}{\FIGUREbox}
}{
  \end{lrbox}

  \centering
  \usebox{\FIGUREbox}

  \caption{\@figureCaption}
  \end{figure}
}

\def\cyrtoc#1{\ifcase #1\or А\or Б\or В\or Г\or Д\or Е\or Ж\or И\or К\or Л\or М\or Н\or П\or Р\or С\or Т\or У\or Ф\or Х\or Ц\or Ш\or Щ\or Э\or Ю\or Я\else \@ctrerr \fi}

\newcommand\originalappendix{}
\let\originalappendix\appendix
\renewcommand{\appendix}{%
  \label{pg:end} % для указания на номер последней страницы
  \originalappendix
  \renewcommand{\thechapter}{\cyrtoc{\value{chapter}}}
  \renewcommand{\appendixname}{Приложение} % Можно не менять и заменить на команду с \MakeUppercase в \renewcommand у меня локально не компилит, а в overleaf все нормально
%  \titleformat{\chapter}{\Large\bfseries\centering}{\MakeUppercase{\chaptertitlename}\ \thechapter}{4pt}{}
%    \titleformat{\chapter}{\Large\bfseries\centering}\chaptertitlename\ \thechapter}{4pt}{}

  \addtocontents{toc}{\protect\value{tocdepth}=\chaptertocdepth}
  \renewcommand\addchaptertocentry[2]{%
    \addtocentrydefault{chapter}{}{\appendixname~##1 ##2}
    % \addtocentrydefault{chapter}{}{\MakeUppercase{\appendixname~##1} ##2}
  }
}
