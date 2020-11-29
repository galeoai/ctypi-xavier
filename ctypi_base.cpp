#include <cstdint>



void ctypi_v3_base(float dx,float dy, uint16_t *im1, uint16_t *im2, int Height, int Width){
    const double c1[7] = {
	0.0116850998497429921230139626686650444753468036651611328125,
	-0.0279730819380002923568717676516826031729578971862792968750,
	0.2239007887600356350166208585505955852568149566650390625000,
	0.5847743866564433234955799889576155692338943481445312500000,
	0.2239007887600356350166208585505955852568149566650390625000,
	-0.0279730819380002923568717676516826031729578971862792968750,
	0.0116850998497429921230139626686650444753468036651611328125 };
    const int ind_off = 3;
    float px2 = 0, py2 = 0, pxy = 0, ABpx = 0, ABpy = 0;
    float tmpABtx = 0, tmpABty = 0, tmp_px = 0, tmp_py = 0;
    for (auto i0 = ind_off; i0 < Width - ind_off; ++i0)
	{
	    for (auto i1 = ind_off; i1 < Height - ind_off; ++i1)
		{
		    tmpABtx = 0; tmpABty = 0; tmp_px = 0; tmp_py = 0;
		    for (auto j0 = -ind_off; j0 <= ind_off; ++j0) {
			tmpABtx += (im1[(i0 + j0) * Height + i1] -
				    im2[(i0 + j0) * Height + i1]) * c1[j0 + ind_off];
		    }
		    for (auto j1 = -ind_off; j1 <= ind_off; ++j1) {
			tmpABty += (im1[(i0)*Height + i1 + j1] -
				    im2[(i0)*Height + i1 + j1]) * c1[j1 + ind_off];
		    }
		    tmp_py = im1[(i0)*Height + i1 + 1]/2 -
			im1[(i0)*Height + i1 - 1]/2;
		    tmp_px = im1[(i0 + 1) * Height + i1]/2 +	 
			im1[(i0 - 1) * Height + i1]/2;	
		    px2 += tmp_px * tmp_px;
		    py2 += tmp_py * tmp_py;
		    pxy += tmp_px * tmp_py;
		    ABpx += tmpABtx * tmp_px;
		    ABpy += tmpABty * tmp_py;
		}
	}
    dx = (py2 * ABpx - pxy * ABpy) / (px2 * py2 - pxy * pxy);
    dy = (px2 * ABpy - pxy * ABpx) / (px2 * py2 - pxy * pxy);
};
