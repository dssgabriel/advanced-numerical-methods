set term svg size 1920,1080
set size ratio 0.8
set output 'plot_weak.svg'
set title 'Weak scalability test for BLAS parallel'
set title font ', 30'

set key left bottom Right

# set key at 2,0.9,0

set key box lw 2 vertical
set key samplen 5 height 0.6 autotitle
set key spacing 1 font ',20'

set xlabel 'Number of processes' font ', 25' offset 0,-1
set xtics font ', 25' offset 0,-0.5

set ylabel 'Efficiency' font ', 25' offset -0.5
set ytics font ', 25' 

set autoscale xfix
set autoscale yfix
set yrange [0:1.05]
# set offsets 0.1, 0.1, 0.05, 0 # (left, right, top, bottom)
set offsets 0.1, 0.1

set style line 1 lt 1 lw 4 ps 3 pt 1

# plot 'plot_strong.dat' using ($3/$2):xticlabel(1) smooth bezier with lines title 'Average efficiency' ls 1, \
#                    '' using ($3/$2):xticlabel(1) lw 50 ps 2.3 title 'Real Efficiency', \
#                    1 title "Ideal Efficiency"

# '' with labels center offset 3.4,.5 notitle

# plot 'plot_strong.dat' using 4:($3/$2):xticlabel(4) with lp ls 1 lc 1 title 'Real Speedup', \
  # 'plot_strong.dat' using ($3/$2):xticlabel($4) with lp ls 1 lc 1 title 'Real Efficiency'

# BLAS1

# 'weak' 'N' 'daxpy base' 'dasum base' 'ddot base' 'dnrm2 base' 'daxpy' 'dasum' 'ddot' 'dnrm2'" >> $f
plot 'weak.data' using 2:($7/$3) with lp ls 1 lc 1 title 'daxpy speedup', \
      '' u 2:($8/$4) with lp ls 1 lc 4 title 'dasum Speedup', \
      '' u 2:($9/$5) with lp ls 1 lc 5 title 'ddot Speedup', \
      '' u 2:($10/$6) with lp ls 1 lc 6 title 'dnrm2 Speedup', \
      1 title "Ideal Speedup" ls 1 lc 2

# BLAS2
# plot 'strong.data' using 2:($3/$4) with lp ls 1 lc 1 title 'dgemv speedup', \
#       '' u 2:($5/$6) with lp ls 1 lc 4 title 'dger Speedup', \
#       ideal(x) title "Ideal Speedup" ls 1 lc 2

# BLAS3
# plot 'strong.data' using 2:($3/$4) with lp ls 1 lc 1 title 'dgemm speedup', \
#       ideal(x) title "Ideal Speedup" ls 1 lc 2
