set term png size 1920,1080 enhanced font "Source Sans Pro,11"
set grid
set size ratio 0.8
set output 'plot_strong.png'
set title 'Strong scalability test for BLAS parallel'
set title font ', 30'

set key left top Right

ideal(x)=x

set key box lw 2 vertical
set key samplen 5 height 0.6 autotitle
set key spacing 1 font ',20'

set xlabel 'Number of processes' font ', 25' offset 0,-1
set xtics font ', 25' offset 0,-0.5

set ylabel 'Speedup' font ', 25' offset -0.5
set ytics font ', 25' 

set logscale x 2
set logscale y 2

set style line 1 lt 1 lw 4 ps 3 pt 1

# BLAS1
#plot 'bench/blas1/strong/blas1_strong.data' using 2:($3/$4) with lp ls 1 lc 1 title 'daxpy speedup', \
#      '' u 2:($5/$6) with lp ls 1 lc 4 title 'ddot speedup', \
#      '' u 2:($7/$8) with lp ls 1 lc 5 title 'dnrm2 speedup', \
#      '' u 2:($9/$10) with lp ls 1 lc 6 title 'dmax speedup', \
#      ideal(x) title "Ideal speedup" ls 1 lc 2

# BLAS3
plot 'bench/blas3/strong/blas3_strong.data' using 2:($3/$4) with lp ls 1 lc 1 title 'dgemm speedup', \
      '' u 2:($5/$6) with lp ls 1 lc 4 title 'dgemm variant speedup', \
      ideal(x) title "Ideal speedup" ls 1 lc 2
