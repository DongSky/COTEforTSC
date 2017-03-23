/**
 * Created by DongSky on 2017/3/18.
 */

public class CvCompare implements Comparable<CvCompare>{
	public double cvAccuracy;
	public int classifierId;
	public CvCompare(double cvAccuracy,int classifierId){
		this.cvAccuracy=cvAccuracy;
		this.classifierId=classifierId;
	}
	
	@Override
	public int compareTo(CvCompare o) {
		if(this.cvAccuracy<o.cvAccuracy)return -1;
		else if(this.cvAccuracy>o.cvAccuracy)return 1;
		else return 0;
	}
}
