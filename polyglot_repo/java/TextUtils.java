package util;

import java.util.Locale;

/** String helpers shared across services. */
public final class TextUtils {

    /** Lowercase and collapse non-alphanumerics into single dashes. */
    public static String slugify(String value) {
        String s = value.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-");
        return s.replaceAll("^-+|-+$", "");
    }

    /** Truncate to at most `max` chars, appending an ellipsis if cut. */
    public static String truncate(String value, int max) {
        if (value.length() <= max) {
            return value;
        }
        return value.substring(0, Math.max(0, max - 1)) + "…";
    }
}
