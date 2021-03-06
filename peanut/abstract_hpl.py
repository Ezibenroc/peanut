import itertools
import os
import random
from .peanut import Job, logger


class AbstractHPL(Job):
    hpl_dir = 'hpl-2.2'
    expfile_sets = {'matrix_size': range(1, 2**30),
                    'block_size': range(1, 2**12),
                    'proc_p': range(1, 2**16),
                    'proc_q': range(1, 2**16),
                    'pfact': range(0, 3),
                    'rfact': range(0, 3),
                    'bcast': range(0, 6),
                    'depth': range(0, 2),
                    'swap': range(0, 3),
                    'mem_align': [4, 8],
                    'process_per_node': range(1, 129),
                    'thread_per_process': range(1, 129),
                    }
    expfile_types = {fact: int for fact in expfile_sets}
    installfile_types = {'trace_execution': bool, 'terminate_early': bool, 'insert_bcast': bool, 'openblas': str}

    @classmethod
    def check_exp(cls, exp):
        for fact, allowed_val in cls.expfile_sets.items():
            if exp[fact] not in allowed_val:
                raise ValueError('Error with exp %s: wrong value for factor %s (%s).' % (exp, fact, exp[fact]))

    def setup(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        self.apt_install(
            'build-essential',
            'zip',
            'make',
            'git',
            'time',
            'hwloc',
            'pciutils',
            'cmake',
            'cpufrequtils',
            'linux-cpupower',
        )
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout=install_options['openblas'])
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')

    @staticmethod
    def generate_hpl_file(*, matrix_size, block_size, proc_p, proc_q, pfact, rfact, bcast, depth, swap, mem_align, **a):
        content = '\n'.join([
            'HPLinpack benchmark input file',
            'Innovative Computing Laboratory, University of Tennessee',
            'HPL.out         output file name (if any)',
            '6               device out (6=stdout,7=stderr,file)',
            '1               # of problems sizes (N)',
            '{matrix_size}          # default: 29 30 34 35  Ns',
            '1               # default: 1            # of NBs',
            '{block_size}    # 1 2 3 4      NBs',
            '0               PMAP process mapping (0=Row-,1=Column-major)',
            '1               # of process grids (P x Q)',
            '{proc_p}        Ps',
            '{proc_q}        Qs',
            '16.0            threshold',
            '1               # of panel fact',
            '{pfact}         PFACTs (0=left, 1=Crout, 2=Right)',
            '1               # of recursive stopping criterium',
            '2               NBMINs (>= 1)',
            '1               # of panels in recursion',
            '2               NDIVs',
            '1               # of recursive panel fact.',
            '{rfact}         RFACTs (0=left, 1=Crout, 2=Right)',
            '1               # of broadcast',
            '{bcast}         BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)',
            '1               # of lookahead depth',
            '{depth}         DEPTHs (>=0)',
            '{swap}          SWAP (0=bin-exch,1=long,2=mix)',
            '{block_size}    swapping threshold',
            '0               L1 in (0=transposed,1=no-transposed) form',
            '0               U  in (0=transposed,1=no-transposed) form',
            '1               Equilibration (0=no,1=yes)',
            '{mem_align}     memory alignment in double (> 0)',
        ])
        return content.format(
                matrix_size=matrix_size,
                block_size=block_size,
                proc_p=proc_p,
                proc_q=proc_q,
                pfact=pfact,
                rfact=rfact,
                bcast=bcast,
                depth=depth,
                swap=swap,
                mem_align=mem_align)

    @staticmethod
    def parse_hpl(stdout):
        lines = stdout.split('\n')
        residual = -1
        for i, line in enumerate(lines):
            if 'Time' in line and 'Gflops' in line:
                result = lines[i + 2].split()
                result = float(result[-2]), float(result[-1])
            if 'FAILED' in line:
                residual = float(line.split()[-3])
                logger.warning('HPL test failed with residual %.2e (should be < 16).' % residual)
            if 'PASSED' in line:
                residual = float(line.split()[-3])
        result = *result, residual
        return result

    @staticmethod
    def fact_design(factors):
        names, values = zip(*factors.items())
        designs = list(itertools.product(*factors.values()))
        for i in range(len(designs)):
            designs[i] = {n: v for n, v in zip(names, designs[i])}
        return designs

    @classmethod
    def gen_exp(cls):
        factors = dict(cls.expfile_sets)
        factors['matrix_size'] = [250000]
        factors['block_size'] = [2**n for n in range(7, 9)]
        factors['proc_p'] = [32]
        factors['proc_q'] = [32]
        factors['process_per_node'] = [32]
        factors['thread_per_process'] = [1]
        factors['mem_align'] = [8]
        factors['rfact'] = [2]
        factors['pfact'] = [1]
        exp = cls.fact_design(factors)
        random.shuffle(exp)
        return exp

    @classmethod
    def gen_medium_exp(cls):
        factors = dict(cls.expfile_sets)
        factors['matrix_size'] = [2**16]
        factors['block_size'] = [2**7]
        factors['proc_p'] = factors['proc_q'] = [4]
        factors['rfact'] = factors['pfact'] = [2]
        factors['mem_align'] = [8]
        factors['swap'] = [1, 2]
        factors['bcast'] = [0, 1, 2, 3]
        factors['depth'] = [1]
        exp = cls.fact_design(factors)
        exp *= 10
        random.shuffle(exp)
        return exp

    @classmethod
    def gen__exp(cls):
        factors = dict(cls.expfile_sets)
#        factors['matrix_size'] = [2**i for i in range(14, 19)]
#        factors['matrix_size'] += [s + s//2 for s in factors['matrix_size'][:-1]]
        factors['matrix_size'] = list(range(300000, 500001, 50000)) # list(range(50000, 250001, 25000))
        factors['block_size'] = [2**7]
        factors['proc_p'] = [32]
        factors['proc_q'] = [32]
        factors['rfact'] = [2]
        factors['pfact'] = [1]
        factors['mem_align'] = [8]
        factors['bcast'] = [2]
        factors['swap'] = [0]
        factors['depth'] = [1]
        factors['process_per_node'] = [32]
        factors['thread_per_process'] = [1]
        exp = cls.fact_design(factors)
        random.shuffle(exp)
        return exp

    hpl_early_termination_patch = r'''
diff --git a/src/pgesv/HPL_pdgesv0.c b/src/pgesv/HPL_pdgesv0.c
index 8bcf71a..7c7f9fc 100644
--- a/src/pgesv/HPL_pdgesv0.c
+++ b/src/pgesv/HPL_pdgesv0.c
@@ -104,6 +104,8 @@ void HPL_pdgesv0
 /* ..
  * .. Executable Statements ..
  */
+   timestamp_t start = get_timestamp();
+   timestamp_t duration;
    if( ( N = A->n ) <= 0 ) return;

 #ifdef HPL_PROGRESS_REPORT
@@ -126,9 +128,12 @@ void HPL_pdgesv0
    for( j = 0; j < N; j += nb )
    {
       n = N - j; jb = Mmin( n, nb );
+      if(j/nb >= 5) {
+        return;
+      }
 #ifdef HPL_PROGRESS_REPORT
       /* if this is process 0,0 and not the first panel */
-      if ( GRID->myrow == 0 && GRID->mycol == 0 && j > 0 )
+      if ( GRID->myrow == 0 && GRID->mycol == 0 && j > 0 )
       {
           time = HPL_timer_walltime() - start_time;
           gflops = 2.0*(N*(double)N*N - n*(double)n*n)/3.0/(time > 0.0 ? time : 1e-6)/1e9;
@@ -138,18 +143,33 @@ void HPL_pdgesv0
 /*
  * Release panel resources - re-initialize panel data structure
  */
+      start = get_timestamp();
       (void) HPL_pdpanel_free( panel[0] );
       HPL_pdpanel_init( GRID, ALGO, n, n+1, jb, A, j, j, tag, panel[0] );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesv0:init", start, duration, 0, NULL);
 /*
  * Factor and broadcast current panel - update
  */
+      start = get_timestamp();
       HPL_pdfact(               panel[0] );
       (void) HPL_binit(         panel[0] );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesv0:factor", start, duration, 0, NULL);
+      start = get_timestamp();
       do
       { (void) HPL_bcast(       panel[0], &test ); }
       while( test != HPL_SUCCESS );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesv0:broadcast", start, duration, 0, NULL);
+      start = get_timestamp();
       (void) HPL_bwait(         panel[0] );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesv0:wait", start, duration, 0, NULL);
+      start = get_timestamp();
       HPL_pdupdate( NULL, NULL, panel[0], -1 );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesv0:update", start, duration, 0, NULL);
 /*
  * Update message id for next factorization
  */
diff --git a/src/pgesv/HPL_pdgesvK2.c b/src/pgesv/HPL_pdgesvK2.c
index 3aa7f2b..f920c7e 100644
--- a/src/pgesv/HPL_pdgesvK2.c
+++ b/src/pgesv/HPL_pdgesvK2.c
@@ -105,6 +105,8 @@ void HPL_pdgesvK2
 /* ..
  * .. Executable Statements ..
  */
+   timestamp_t start = get_timestamp();
+   timestamp_t duration;
    mycol = GRID->mycol; npcol        = GRID->npcol;
    depth = ALGO->depth; HPL_pdupdate = ALGO->upfun;
    N     = A->n;        nb           = A->nb;
@@ -151,19 +153,31 @@ void HPL_pdgesvK2
 /*
  * Factor and broadcast k-th panel
  */
+      start = get_timestamp();
       HPL_pdfact(         panel[k] );
       (void) HPL_binit(   panel[k] );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesK2:factor", start, duration, 0, NULL);
+      start = get_timestamp();
       do
       { (void) HPL_bcast( panel[k], &test ); }
       while( test != HPL_SUCCESS );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesvK2:broadcast", start, duration, 0, NULL);
+      start = get_timestamp();
       (void) HPL_bwait(   panel[k] );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesvK2:wait", start, duration, 0, NULL);
 /*
  * Partial update of the depth-k-1 panels in front of me
  */
       if( k < depth - 1 )
       {
          nn = HPL_numrocI( jstart-j, j, nb, nb, mycol, 0, npcol );
+         start = get_timestamp();
          HPL_pdupdate( NULL, NULL, panel[k], nn );
+         duration = get_timestamp() - start;
+         record_measure(__FILE__, __LINE__, "HPL_pdgesvK2:update", start, duration, 0, NULL);
       }
    }
 /*
@@ -172,9 +186,12 @@ void HPL_pdgesvK2
    for( j = jstart; j < N; j += nb )
    {
       n = N - j; jb = Mmin( n, nb );
+      if((j-jstart)/nb >= 5) {
+        return;
+      }
 #ifdef HPL_PROGRESS_REPORT
       /* if this is process 0,0 and not the first panel */
-      if ( GRID->myrow == 0 && mycol == 0 && j > 0 )
+      if ( GRID->myrow == 0 && mycol == 0 && j > 0 )
       {
           time = HPL_timer_walltime() - start_time;
           gflops = 2.0*(N*(double)N*N - n*(double)n*n)/3.0/(time > 0.0 ? time : 1e-6)/1e9;
@@ -191,15 +208,24 @@ void HPL_pdgesvK2
       if( mycol == icurcol )
       {
          nn = HPL_numrocI( jb, j, nb, nb, mycol, 0, npcol );
+         start = get_timestamp();
          for( k = 0; k < depth; k++ )   /* partial updates 0..depth-1 */
             (void) HPL_pdupdate( NULL, NULL, panel[k], nn );
+         duration = get_timestamp() - start;
+         record_measure(__FILE__, __LINE__, "HPL_pdgesK2:update", start, duration, 0, NULL);
+         start = get_timestamp();
          HPL_pdfact(       panel[depth] );    /* factor current panel */
+         duration = get_timestamp() - start;
+         record_measure(__FILE__, __LINE__, "HPL_pdgesK2:factor", start, duration, 0, NULL);
       }
       else { nn = 0; }
           /* Finish the latest update and broadcast the current panel */
+      start = get_timestamp();
       (void) HPL_binit( panel[depth] );
       HPL_pdupdate( panel[depth], &test, panel[0], nq-nn );
       (void) HPL_bwait( panel[depth] );
+      duration = get_timestamp() - start;
+      record_measure(__FILE__, __LINE__, "HPL_pdgesK2:update", start, duration, 0, NULL);
 /*
  * Circular  of the panel pointers:
  * xtmp = x[0]; for( k=0; k < depth; k++ ) x[k] = x[k+1]; x[d] = xtmp;
@@ -216,6 +242,7 @@ void HPL_pdgesvK2
 /*
  * Clean-up: Finish updates - release panels and panel list
  */
+   start = get_timestamp();
    nn = HPL_numrocI( 1, N, nb, nb, mycol, 0, npcol );
    for( k = 0; k < depth; k++ )
    {
@@ -223,6 +250,8 @@ void HPL_pdgesvK2
       (void) HPL_pdpanel_disp(  &panel[k] );
    }
    (void) HPL_pdpanel_disp( &panel[depth] );
+   duration = get_timestamp() - start;
+   record_measure(__FILE__, __LINE__, "HPL_pdgesK2:update", start, duration, 0, NULL);

    if( panel ) free( panel );
 /*
'''

    first_bcast_trace_patch = r'''
diff --git a/src/pgesv/HPL_pdgesv.c b/src/pgesv/HPL_pdgesv.c
index af46ee6..c5574c1 100644
--- a/src/pgesv/HPL_pdgesv.c
+++ b/src/pgesv/HPL_pdgesv.c
@@ -94,6 +94,8 @@ void HPL_pdgesv
 /* ..
  * .. Executable Statements ..
  */
+   timestamp_t start = get_timestamp();
+   timestamp_t duration;
    if( A->n <= 0 ) return;

    A->info = 0;
@@ -110,6 +112,8 @@ void HPL_pdgesv
  * Solve upper triangular system
  */
    if( A->info == 0 ) HPL_pdtrsv( GRID, A );
+   duration = get_timestamp() - start;
+   record_measure(__FILE__, __LINE__, "HPL_pdgesv", start, duration, 0, NULL);
 /*
  * End of HPL_pdgesv
  */
diff --git a/src/pgesv/HPL_pdupdateTT.c b/src/pgesv/HPL_pdupdateTT.c
index 57444bc..b990903 100644
--- a/src/pgesv/HPL_pdupdateTT.c
+++ b/src/pgesv/HPL_pdupdateTT.c
@@ -113,6 +113,8 @@ void HPL_pdupdateTT
 /* ..
  * .. Executable Statements ..
  */
+   timestamp_t start_pduptate = get_timestamp();
+   timestamp_t duration;
 #ifdef HPL_DETAILED_TIMING
    HPL_ptimer( HPL_TIMING_UPDATE );
 #endif
@@ -125,18 +127,26 @@ void HPL_pdupdateTT
    {
       if( PBCST != NULL )
       {
+         timestamp_t start_bcast = get_timestamp();
          do { (void) HPL_bcast( PBCST, IFLAG ); }
          while( *IFLAG != HPL_SUCCESS );
+         timestamp_t duration = get_timestamp() - start_bcast;
+         record_measure(__FILE__, __LINE__, "first_bcast", start_bcast, duration, 0, NULL);
       }
 #ifdef HPL_DETAILED_TIMING
       HPL_ptimer( HPL_TIMING_UPDATE );
 #endif
+      duration = get_timestamp() - start_pduptate;
+      record_measure(__FILE__, __LINE__, "HPL_pdupdateTT", start_pduptate, duration, 0, NULL);
       return;
    }
 /*
  * Enable/disable the column panel probing mechanism
  */
+   timestamp_t start_bcast2 = get_timestamp();
    (void) HPL_bcast( PBCST, &test );
+   duration = get_timestamp() - start_bcast2;
+   record_measure(__FILE__, __LINE__, "second_bcast", start_bcast2, duration, 0, NULL);
 /*
  * 1 x Q case
  */
@@ -437,6 +447,8 @@ void HPL_pdupdateTT
 #ifdef HPL_DETAILED_TIMING
    HPL_ptimer( HPL_TIMING_UPDATE );
 #endif
+   duration = get_timestamp() - start_pduptate;
+   record_measure(__FILE__, __LINE__, "HPL_pdupdateTT", start_pduptate, duration, 0, NULL);
 /*
  * End of HPL_pdupdateTT
  */
'''

    hpl_bcast_patch = r'''
diff --git a/testing/ptest/HPL_pdtest.c b/testing/ptest/HPL_pdtest.c
index 33b11ac..dea0d93 100644
--- a/testing/ptest/HPL_pdtest.c
+++ b/testing/ptest/HPL_pdtest.c
@@ -197,7 +197,14 @@ void HPL_pdtest
    HPL_ptimer_boot(); (void) HPL_barrier( GRID->all_comm );
    time( &current_time_start );
    HPL_ptimer( 0 );
+   int n = 12;
+   MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
+   timestamp_t start = get_timestamp();
+   record_measure(__FILE__, __LINE__, "smpi_marker", start, 0, 0, NULL);
    HPL_pdgesv( GRID, ALGO, &mat );
+   timestamp_t end = get_timestamp();
+   record_measure(__FILE__, __LINE__, "smpi_marker", end, 0, 0, NULL);
+   MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    HPL_ptimer( 0 );
    time( &current_time_end );
 #ifdef HPL_CALL_VSIPL
    '''

    @property
    def makefile(self):
        return '''
# -*- makefile -*-
#
#  -- High Performance Computing Linpack Benchmark (HPL)
#     HPL - 2.0 - September 10, 2008
#     Antoine P. Petitet
#     University of Tennessee, Knoxville
#     Innovative Computing Laboratory
#     (C) Copyright 2000-2008 All Rights Reserved
#
#  -- Copyright notice and Licensing terms:
#
#  Redistribution  and  use in  source and binary forms, with or without
#  modification, are  permitted provided  that the following  conditions
#  are met:
#
#  1. Redistributions  of  source  code  must retain the above copyright
#  notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce  the above copyright
#  notice, this list of conditions,  and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
#  3. All  advertising  materials  mentioning  features  or  use of this
#  software must display the following acknowledgement:
#  This  product  includes  software  developed  at  the  University  of
#  Tennessee, Knoxville, Innovative Computing Laboratory.
#
#  4. The name of the  University,  the name of the  Laboratory,  or the
#  names  of  its  contributors  may  not  be used to endorse or promote
#  products  derived   from   this  software  without  specific  written
#  permission.
#
#  -- Disclaimer:
#
#  THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
#  OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
#  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ######################################################################
#
# ----------------------------------------------------------------------
# - shell --------------------------------------------------------------
# ----------------------------------------------------------------------
#
SHELL        = /bin/sh
#
CD           = cd
CP           = cp -f
LN_S         = ln -f -s
MKDIR        = mkdir -p
RM           = /bin/rm -f
TOUCH        = touch
#
# ----------------------------------------------------------------------
# - Platform identifier ------------------------------------------------
# ----------------------------------------------------------------------
#
ARCH         = $(arch)
#
# ----------------------------------------------------------------------
# - HPL Directory Structure / HPL library ------------------------------
# ----------------------------------------------------------------------
#
TOPdir       = %s
INCdir       = $(TOPdir)/include
BINdir       = $(TOPdir)/bin/$(ARCH)
LIBdir       = $(TOPdir)/lib/$(ARCH)
#
HPLlib       = $(LIBdir)/libhpl.a
#
# ----------------------------------------------------------------------
# - MPI directories - library ------------------------------------------
# ----------------------------------------------------------------------
# MPinc tells the  C  compiler where to find the Message Passing library
# header files,  MPlib  is defined  to be the name of  the library to be
# used. The variable MPdir is only used for defining MPinc and MPlib.
#
MPdir        =
MPinc        =
MPlib        =
#
# ----------------------------------------------------------------------
# - Linear Algebra library (BLAS or VSIPL) -----------------------------
# ----------------------------------------------------------------------
# LAinc tells the  C  compiler where to find the Linear Algebra  library
# header files,  LAlib  is defined  to be the name of  the library to be
# used. The variable LAdir is only used for defining LAinc and LAlib.
#
LAdir        = /tmp/lib
LAinc        =
LAlib        = /tmp/lib/libopenblas.so
#
# ----------------------------------------------------------------------
# - F77 / C interface --------------------------------------------------
# ----------------------------------------------------------------------
# You can skip this section  if and only if  you are not planning to use
# a  BLAS  library featuring a Fortran 77 interface.  Otherwise,  it  is
# necessary  to  fill out the  F2CDEFS  variable  with  the  appropriate
# options.  **One and only one**  option should be chosen in **each** of
# the 3 following categories:
#
# 1) name space (How C calls a Fortran 77 routine)
#
# -DAdd_              : all lower case and a suffixed underscore  (Suns,
#                       Intel, ...),                           [default]
# -DNoChange          : all lower case (IBM RS6000),
# -DUpCase            : all upper case (Cray),
# -DAdd__             : the FORTRAN compiler in use is f2c.
#
# 2) C and Fortran 77 integer mapping
#
# -DF77_INTEGER=int   : Fortran 77 INTEGER is a C int,         [default]
# -DF77_INTEGER=long  : Fortran 77 INTEGER is a C long,
# -DF77_INTEGER=short : Fortran 77 INTEGER is a C short.
#
# 3) Fortran 77 string handling
#
# -DStringSunStyle    : The string address is passed at the string loca-
#                       tion on the stack, and the string length is then
#                       passed as  an  F77_INTEGER  after  all  explicit
#                       stack arguments,                       [default]
# -DStringStructPtr   : The address  of  a  structure  is  passed  by  a
#                       Fortran 77  string,  and the structure is of the
#                       form: struct {char *cp; F77_INTEGER len;},
# -DStringStructVal   : A structure is passed by value for each  Fortran
#                       77 string,  and  the  structure is  of the form:
#                       struct {char *cp; F77_INTEGER len;},
# -DStringCrayStyle   : Special option for  Cray  machines,  which  uses
#                       Cray  fcd  (fortran  character  descriptor)  for
#                       interoperation.
#
F2CDEFS      =
#
# ----------------------------------------------------------------------
# - HPL includes / libraries / specifics -------------------------------
# ----------------------------------------------------------------------
#
HPL_INCLUDES = -I$(INCdir) -I$(INCdir)/$(ARCH) $(LAinc) $(MPinc)
HPL_LIBS     = $(HPLlib) $(LAlib) $(MPlib) -lm
#
# - Compile time options -----------------------------------------------
#
# -DHPL_COPY_L           force the copy of the panel L before bcast;
# -DHPL_CALL_CBLAS       call the cblas interface;
# -DHPL_CALL_VSIPL       call the vsip  library;
# -DHPL_DETAILED_TIMING  enable detailed timers;
#
# By default HPL will:
#    *) not copy L before broadcast,
#    *) call the Fortran 77 BLAS interface
#    *) not display detailed timing information.
#
# The last option, -DSMPI_OPTIMIZATION, is only used in the simulation. It should have *no* impact on the real
# execution.
HPL_OPTS     = -DHPL_CALL_CBLAS -DHPL_NO_MPI_DATATYPE -DHPL_USE_GETTIMEOFDAY -DSMPI_OPTIMIZATION
#
# ----------------------------------------------------------------------
#
HPL_DEFS     = $(F2CDEFS) $(HPL_OPTS) $(HPL_INCLUDES)
#
# ----------------------------------------------------------------------
# - Compilers / linkers - Optimization flags ---------------------------
# ----------------------------------------------------------------------
#
CC           = mpicc
CCNOOPT      = $(HPL_DEFS)
CCFLAGS      = $(HPL_DEFS) -fomit-frame-pointer -O3 -funroll-loops -W -Wall $(shell dpkg-buildflags --get CFLAGS)
#
LINKER       = mpicc
LINKFLAGS    = $(CCFLAGS) $(shell dpkg-buildflags --get LDFLAGS)
#
ARCHIVER     = ar
ARFLAGS      = r
RANLIB       = echo
#
# ----------------------------------------------------------------------
''' % os.path.join(self.nodes.working_dir, self.hpl_dir)
