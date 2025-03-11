Phase from Envelope Detector. Consider a setup in
which an AP transmits RF signals using two of its antennas
as shown in Fig. 6. The two transmissions will travel diferent
distances to the receiver and therefore combine at the insect
with a phase diference corresponding to this distance difer-
ence. The small antenna at the insect receives the combined
RF signals and the envelope detector outputs its amplitude.
The key insight here is that the amplitude depends upon the
phase diference at which the two transmitted signals com-
bine at the receiver. For example, if the transmitted signals
are perfectly in-phase they will add constructively giving
the maximum amplitude; in contrast a phase diference of π
will cause the two signals to cancel each other completely.
Our key insight is that by intentionally introducing an
additional phase diference between the two AP antennas,
we can create amplitude changes at the receiver. We can
analyze these changes to estimate the phase corresponding
to the angle of the insect from the AP.
To explain this in more detail, consider x1(t) and x2(t) to
be the signals transmitted from the two antennas. Let us set
both these signals to x(t) Ae jωt . Assuming no multipath
which we will discuss later, the signal at the receiver envelope
detector can now be written as:

y(t)= $ |ax_ {1} $ (t)+ $ ae^ {j\phi } $ $ x_ {2} $ (t)|=aA( $ \sqrt {2+2\cos (\phi )} $ )

Here a is the signal attenuation, and ϕ is the phase difer-
ence between the two paths. Note that the amplitude attenu-
ation diference between the two signals is negligible since
the separation between the two antennas is small compared
to the distance between the AP and bee.
Now if the AP intentionally introduces a phase diference
of θ on the second transmit antenna, i.e., 

 $ x_ {2} $ (t)=x(t) $ e^ {-j\theta } $ ,

,
the signal at the receiver can be written as,


y(t)=|ax(t)+ $ ae^ {j(\phi -\theta )} $ x(t)|=aA( $ \sqrt {2+2\cos (\phi -\theta )} $ )

The maximum value for the above equation happens when
ˆ
ϕ− θ 0 mod 2π . Hence, to get an estimation
ϕ of ϕ, we
can let the AP sweep θ from−π to π at a constant rate.
At the receiver side, the envelope detector simply samples
the amplitudes corresponding to each of the θ s. It then gets
the sample with the maximum amplitude and infers θmax
ˆ
from the time of that sample. Now, the
ϕ we are interested
ˆ
in is simply θmax . The angle
Θ of arrival of the receiver
corresponding to the two transmit antennas at the AP can be
ˆ
ˆ
derived by
ϕ d sin(
Θ) where d is the distance between the
two transmit antennas in radians. In our design, this distance
between transmit antennas is set to half a wavelength.

A key consideration in the above design is that as the dis-
tance increases, noise afects the signal quality. We mitigate
the efect of this noise by sampling for a longer time, i.e., the
AP dwells on each phase diference θ for a longer duration.
The upper bound on this duration however is determined
by the motion of the bee. Our empirical results found that a
duration of 50 ms per sweep across all the angles, is a good
trade-of between the noise level and motion tolerance.



