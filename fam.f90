!---------------------------------------------------------------------- 
!---------------------------------------------------------------------- 
!   fam : fattened arnold map (issues:replaced tn+1, unsure of dfdp)
!---------------------------------------------------------------------- 
!---------------------------------------------------------------------- 

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP) 
!     ---------- ---- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

       F(1)=PAR(4)*(1-PAR(1))+PAR(1)*U(1)+PAR(2)*SIN((2*3.14159265359&
       *(U(1)+U(2))))/PAR(3)
       F(2)=U(1)+U(2)

      IF(IJAC.EQ.0)RETURN 

       DFDU(1,1)=PAR(1)+2*PAR(2)*3.14159265359*COS(2*3.14159265359*&
       (U(1)+U(2)))/PAR(3) 
       DFDU(1,2)=2*PAR(2)*3.14159265359*COS(2*3.14159265359*(U(1)+&
       U(2)))/PAR(3)
       DFDU(2,1)=U(2) 
       DFDU(2,2)=U(1) 

      IF(IJAC.EQ.1)RETURN 

       DFDP(1,1)=PAR(4)+U(1) 
       DFDP(2,1)=0.0 
       DFDP(1,2)=SIN((2*3.14159265359*(U(1)+U(2))))/PAR(4) 
       DFDP(2,2)=0
       DFDP(1,3)=LOG(PAR(3))*SIN((2*3.14159265359*(U(1)+U(2))))/PAR(4)
       DFDP(2,3)=0
       DFDP(1,4)=1-PAR(1)
       DFDP(2,4)=0

      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)
!     ---------- ----- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

       PAR(1)=0.0 
       PAR(2)=0.5 
       PAR(3)=24
       PAR(4)=1

       U(1)=0.0 
       U(2)=0.0 

      END SUBROUTINE STPNT

      SUBROUTINE BCND 
      END SUBROUTINE BCND

      SUBROUTINE ICND 
      END SUBROUTINE ICND

      SUBROUTINE FOPT 
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
