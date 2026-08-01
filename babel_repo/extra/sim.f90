! Physics helpers.
module physics
contains
  ! Kinetic energy in joules.
  function kinetic(m, v) result(e)
    real, intent(in) :: m, v
    real :: e
    e = 0.5 * m * v * v
  end function kinetic

  subroutine report(e)
    real, intent(in) :: e
    print *, e
  end subroutine report
end module physics
