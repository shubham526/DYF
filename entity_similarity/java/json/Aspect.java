package json;

import org.jetbrains.annotations.NotNull;
import org.json.JSONException;
import org.json.JSONObject;

public class Aspect {
    private final String aspectId;
    private final String aspectName;
    private final Location location;
    private final AnnotatedText aspectContent;

    public Aspect(@NotNull JSONObject jsonObject) throws JSONException {
        this.aspectId = jsonObject.getString("aspect_id");
        this.aspectName = jsonObject.getString("aspect_name");
        this.location = new Location(jsonObject.getJSONObject("location"));
        this.aspectContent = new AnnotatedText(jsonObject.getJSONObject("aspect_content"));
    }

    @Override
    public String toString() {
        return "Aspect{" +
                "aspectId='" + aspectId + '\'' +
                ", aspectName='" + aspectName + '\'' +
                ", location=" + location +
                ", aspectContent=" + aspectContent +
                '}';
    }

    public String getAspectId() {
        return aspectId;
    }

    public String getAspectName() {
        return aspectName;
    }

    public Location getLocation() {
        return location;
    }

    public AnnotatedText getAspectContent() {
        return aspectContent;
    }
}
