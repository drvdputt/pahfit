import numpy.ma as ma
from astropy.table import MaskedColumn
from astropy.table.pprint import TableFormatter


# * Special table formatting for bounded (val, min, max) values
def fmt_func(fmt):
    def _fmt(v):
        try:
            if ma.is_masked(v[0]):
                return "  masked  "
            if ma.is_masked(v[1]):
                return f"{v[0]:{fmt}} (Fixed)"
            return f"{v[0]:{fmt}} ({v[1]:{fmt}}, {v[2]:{fmt}})"
        except Exception as e:
            # print(f"problem in _fmt: v = {v}, fmt = {fmt}, type(fmt) = {type(fmt)}")
            # raise e
            return f"{v:0.4g}"

    return _fmt


class BoundedMaskedColumn(MaskedColumn):
    """Masked column which can be toggled to group rows into one item
    for formatting.  To be set as Table's `MaskedColumn'.
    """

    _omit_shape = False

    @property
    def shape(self):
        # return (sh[0],) if self._omit_shape and len(sh) > 1 else sh
        return super().shape

    def is_fixed(self):
        return ma.getmask(self)[:, 1:].all(1)


class BoundedParTableFormatter(TableFormatter):
    """Format bounded parameters.
    Bounded parameters are 3-field structured arrays, with fields
    'var', 'min', and 'max'.  To be set as Table's `TableFormatter'.
    """

    # def _pformat_table(self, table, *args, **kwargs):
    #     #return super()._pformat_table(table, *args, **kwargs)
    #     bpcols = []
    #     try:
    #         colsh = [(col, col.shape) for col in table.columns.values()]
    #         for col, sh in colsh:
    #             if len(sh) == 2 and sh[1] == 3:
    #                 bpcols.append((col, col.info.format))
    #                 print("before", col.name, sh, col.info.format)
    #                 col.info.format = fmt_func("0.4g")
    #                 print("after", col.name, sh, col.info.format)
    #         BoundedMaskedColumn._omit_shape = True
    #         return super()._pformat_table(table, *args, **kwargs)
    #     finally:
    #         BoundedMaskedColumn._omit_shape = False
    #         for col, fmt in bpcols:
    #             print("finally", col.name, fmt)
    #             col.info.format = fmt

    def _pformat_col(
        self,
        col,
        max_lines=None,
        **kwargs,
    ):
        default_strings, outs = super()._pformat_col(col, max_lines, **kwargs)
        has_bounds = len(col.shape) == 2 and col.shape[1] == 3
        if not has_bounds:
            strings = default_strings
        else:
            # copy the header strings
            strings = default_strings[: outs["n_header"]]

            # modify output of the numerical lines
            # data index = line index - n_header
            after_ellipsis = False
            data_strings = default_strings[outs["n_header"] :]
            for i, line in enumerate(data_strings):
                if "..." in line or "table" in line:
                    # generic ellipsis line
                    strings.append(line)
                    after_ellipsis = True
                elif "-- .. --" in line:
                    # all masked
                    strings.append(line.replace("-- .. --", "--"))
                elif ".. --" in line:
                    # value but masked bound
                    strings.append(line.replace(".. --", "(fixed)"))
                else:
                    # deal with case where a "..." row is present, by counting from the end
                    # show -1 if at the last data string
                    if after_ellipsis:
                        d_from_end = len(data_strings) - i
                        if outs['show_length']:
                            # need to add 1 because "length = xx rows" is the last row in this case
                            d_from_end -= 1
                        row_index = -d_from_end
                    else:
                        row_index = i

                    strings.append(fmt_func(col.info.format or "0.4g")(col[row_index]))

        return strings, outs
