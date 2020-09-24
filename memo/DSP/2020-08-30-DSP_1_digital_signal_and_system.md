<!-- ---
layout: post
title: < DSP 정리 > 1. Digital Signal과 Digital System
category: dev
permalink: /DSP/:year/:month/:day/:title/
tags: dev dsp DSP정리시리즈
comments: true
--- -->

TO BE CONTINUE...
=================

Reference
---------

본 내용은 공부한 내용들을 다시 한번 정리하는 마음으로 제작했습니다. <br> 혹시 잘못된 내용이나 참조에 문제가 있는 경우 바로 조치하겠습니다. <br>

1.	Oppenheim A. V. Schafer R. W. - Discrete-Time Signal Processing 3rd Edition (Prentice-Hall Signal Processing Series) - 2010
2.	[디지털 신호처리 Jhmoon93@gmail.com.](https://slidesplayer.org/slide/14895988/)
3.	[DSP 한글 정리 페이지](https://kascia.github.io/dsp/)
4.	[Jaejun Yoo's Playground blog](http://jaejunyoo.blogspot.com/2019/05/signal-processing-for-communications.html)
5.	[KOCW 디지털신호처리 전남대학교 김진영교수님](http://www.kocw.net/home/search/kemView.do?kemId=153546)
6.	[KOCW 디지털신호처리 한양대학교 양성일교수님](http://www.kocw.or.kr/home/cview.do?mty=p&kemId=1223167)
7.	[M I P Lab](https://sites.google.com/site/miplaboratory/lecture/digital-signal-processing)
8.	[어썸 naver blog](http://blog.naver.com/PostView.nhn?blogId=cutterpoong&logNo=30911692)
9.	[숭실대 Power System Lab](http://power10.ssu.ac.kr/xe/?mid=DSP&page=1&listStyle=webzine&document_srl=144774)
10.	[상지대학교 DSP강좌](https://www.sangji.ac.kr/cmm/fms/FileDown.do?atchFileId=FILE_000000000021028&fileSn=0)

---

목차 <br> 1. [신호와 시스템](#신호와-시스템) <br> 1) 신호의 정의<br> 2) 예시<br> 3) 신호의 분류<br> 4) 시스템의 정의<br> 5) 신호 처리의 정의<br> 6) 신호 처리의 필요성<br><br> 2. 디지털 신호<br> 1) 디지털 신호의 정의<br> 2) 디지털 신호의 장점<br> 3) 디지털 신호 처리의 정의<br>

<br>

---

신호와 시스템
-------------

### 신호(Signal)란?

-	**신호**란 어떠한 **매체**를 통하여 **정보를 전달**하기 위한 **물리적인 파형**을 말합니다
-	보통 **시간**과 **공간** 정보와 같은 값을 신호의 **독립 변수(Independent Variable)**로 사용한다.

예 전압, 온도, 압력, ... 음성신호, 오디오신호, 영상신호, 레이더신호
