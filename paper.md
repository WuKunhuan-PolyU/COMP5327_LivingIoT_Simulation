




The key insight is that, when M
j 2 aj <a1, which is true 
in most line-of-sight scenarios like the farm, the error in our estimate of the angle with respect to the AP, which is |
ϕ−ϕ1 |, is bounded. Moreover, this error decreases linearly as the 
number of antennas increases. To verify this intuition, we 
perform a simulation where we compute this error by chang-
ing the number of antennas. We repeat this for increasing
multipath ratios R, ratio of the sum of the amplitudes of all
NLOS paths with respect to the amplitude of the LOS path.
Assuming that the angles of indirect paths are uniformly dis-
tributed, Fig. 7 shows the mean error as a function of these
two parameters. The plot shows that the error is less than
10◦ when using four antennas even if the total amplitude of
all NLOS path is 60% of the amplitude of LOS path. With ive
antennas, we can get a similar error even when this ratio is
close to 0.95. This shows that by increasing the number of
antennas at the AP, we can reduce the error due to NLOS
paths and achieve an accurate angle estimation.

For our experiments, each AP
consists of one USRP-N210 connected to a four way power
splitter followed by three phase shifters, each of which in-
troduces a phase shift controlled by an NI myDAQ digital-
analog converter. Along with the original signal, the four
outputs are ampliied to 28 dBm by a Qorvo RF5110G power
ampliier and then connected to four 2 dBi monopole anten-
nas separated by 12 cm each. The cable lengths are carefully
calibrated so that no extra phase ofset is introduced.

------------



# Living IoT Paper

Living IoT’s localization system employs APs which transmit
RF signals in the 900 MHz ISM band. At the insect-mounted
receiver, due to the power and size constraints, living IoT uses
a passive envelope detector connected to a small antenna to
receive only the amplitude of the RF signals broadcast by the
APs. Unlike an active radio that gives both amplitude and
phase, the envelope detector only provides the amplitude
of the signal but can do so with small, zero-power passive
hardware components [37, 64]. The phase of the signal is
however essential to achieve wireless localization.
To address this, we irst extract the phase diference of the
signals from each of the transmit antennas of an AP using the
amplitude output by the envelope detector. We then use this
phase to compute the angle of the insect with respect to that
AP. By using the angles from two APs we can identify the
2D location. Typically bees only ly a few meters above the
ground and hence 3D localization is not essential. However,
the technique presented in this paper can be generalized to
achieve 3D localization by adding an additional AP. Next,
we describe each of these techniques in further detail.

## Phase from Envelope Detector. 

Consider a setup in
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

## Addressing Multi-path

The above discussion assumes
ˆ
that the phase diference
ϕ estimated at the insect can be
translated into angle of arrival using amplitude variations.
We however need to address the potential amplitude vari-
ations due to multipath. Unlike systems like Wi-Fi which
operate indoors with mostly no LOS path, our system is de-
signed for outdoor farm use in the natural habitat of insects.
For example, in deployment scenarios such as open ields and
farms there is a direct strong line of sight. However, we still
need to account for the amplitude variations that result from
the other paths constructively and destructively interfering
with the dominant direct line of sight path. To this end, we
utilize more than two antennas per AP to reduce the error
in the angle caused by multipath.
Formally, suppose we use N antennas separated by half a
wavelength each. Similar to the above two-antenna scenario,
we introduce a phase ofset of θi for the ith antenna. 

We solve for θmax using the same procedure as in the 2-
ˆ
antenna case and estimate
ϕ θmax , where the amplitude
output by the envelope detector has the maximum value.
The key insight is that, when M
j 2 aj <a1, which is true
in most line-of-sight scenarios like the farm, the error in our
ˆ
estimate of the angle with respect to the AP, which is |
ϕ−ϕ1 |,
is bounded. Moreover, this error decreases linearly as the
number of antennas increases. To verify this intuition, we 
perform a simulation where we compute this error by chang-
ing the number of antennas. We repeat this for increasing
multipath ratios R, ratio of the sum of the amplitudes of all
NLOS paths with respect to the amplitude of the LOS path.
Assuming that the angles of indirect paths are uniformly dis-
tributed, Fig. 7 shows the mean error as a function of these
two parameters. The plot shows that the error is less than
10◦ when using four antennas even if the total amplitude of
all NLOS path is 60% of the amplitude of LOS path. With ive
antennas, we can get a similar error even when this ratio is
close to 0.95. This shows that by increasing the number of
antennas at the AP, we can reduce the error due to NLOS
paths and achieve an accurate angle estimation. 



## Leveraging insect motion.

We also leverage insect mo-
tion to reduce the efects of multipath and reduce the proba-
bility of small scale fading. This is speciically useful when
the bee is in motion. Since the typical speeds when the bee
is in motion are less than 10 m/s, the bee does not move
by more than a meter between consecutive 50 ms durations
where our AP cycles across the phase values. Despite in-
troducing slight errors because of Doppler efects, our algo-
rithm utilizes motion to improve the accuracy by leveraging
spatial diversity: the multipath combination can be signii-
cantly diferent for even a small displacement. Hence, we use
exponential smoothing to temporally average consecutive
measurements which yields a more accurate result. Formally,
the inal angle of arrival Θt of each bee is calculated as:
ˆ
Θt ηΘt−1 (1− η) sin−1(
ϕt /π ), where η is the smoothing
constant which we set to 0.8. Note that the computation-
ally expensive sin−1 operation can be oloaded to the access
point, as shown in Algorithm 1.



## 2D localization of insects. 

To estimate the 2D location
of the insect, we employ two APs with four antennas each.
Separating the two APs and placing them perpendicular to
each other gives the best 2D accuracy. Given the known loca-
tions of the two APs, the two angles computed with respect
to each AP, uniquely identiies the 2D location. The bees can
either store these two angles or can also calculate their 2D 
location of using the intersection of the two separate angles
of arrival. This intersection procedure can be implemented
using a look-up table to minimize the required computation.
Speciically, the two APs intermittently transmit their
sweep signal one after another. They are coarsely synchro-
nized using TDMA so that no two sweeps will interfere with
each other. We transmit two predeined orthogonal pream-
bles using ON-OFF keying, [1,0,1,0,1,0,1,0] and [1,1,0,0,1,1,0,0]
to identify each AP. The preambles are transmitted before
every sweep of each AP respectively so that the bee can ind
the start of each sweep eiciently. The receiver irst detects
the preamble using a simple state machine, then runs the
algorithm twice to get the two angles. Our pseudo-code is
presented in Algorithm 2. We note that the computation for
our receiver algorithm scales linearly with the sampling rate
(which is in the order of kHz), and only simple arithmetic
operations are involved. This makes it eicient enough to
run on our microcontroller platform.
A key consideration while increasing the number of APs
is that the transmissions from each of them have to be time-
multiplexed. This increases the delay required to compute
the location value which can be challenging especially when
the bee is highly mobile. Given that each AP sweeps across
various phase values in 50 ms segments, the delay to com-
pute the 2D location is around 100 ms. Assuming a speed of
3 m/s, this translates to a motion of around 30 cm which is
within the error of our location estimates and hence does
not signiicantly afect our accuracies.


















for APi in {AP1,AP2 } do
APi transmit preamblei
for Θ−π /2 to π /2 by δ do
for j 2 to 4 do
set the phase shift of the jth phase shifter to
(j− 1) ∗ π ∗ sin(Θ)
end
sleep for (T− Tpr eambl e )/π ∗ δ
end
end

Algorithm 1: Pseudo-code at the AP.





while True do
receive and store amplitude samples during the last
3T time into memory as S
if preamble1 detected in S1..2T located at i then
p1 arдmaxj ∈(Tpreamble,T )Si j
anдle1 (p1−Tpr eambl e )/(T−Tpr eambl e )∗π−π /2
locate preamble2 in Si T ..3T at i ′
p2 arдmaxj ∈(Tpreamble,T )Si′ j
anдle2 (p2−Tpr eambl e )/(T−Tpr eambl e )∗π−π /2
(If Euclidean position is needed) get x and y
from anдle1 and anдle2 using look-up table
output (anдle1,anдle2) or (x, y)
end
end
Algorithm 2: Pseudo-code of Living IoT platform.


